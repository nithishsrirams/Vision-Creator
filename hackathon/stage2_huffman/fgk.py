"""
Adaptive Huffman Encoding — FGK Algorithm
(Faller, Gallager, Knuth, 1985)

Quick background on why "adaptive":

Regular Huffman builds a frequency table over the whole input first,
then encodes. That means two passes over the data, and you have to
transmit the entire tree as a header so the decoder can reconstruct it.

Adaptive Huffman does it in one pass. The encoder and decoder both
start with the same empty tree and update it identically as each
symbol is processed. No header needed — the tree is implicit.

The FGK algorithm maintains one invariant throughout: the sibling
property. Every node's weight must be >= its right neighbor's weight
when nodes are ordered left-to-right by weight. When a symbol is
encoded, we update weights and swap nodes to restore this property.
That swapping is what keeps the tree optimal at every step.
"""


# ---------------------------------------------------------------------------
# Node — one cell in the Huffman tree
# ---------------------------------------------------------------------------

class Node:
    """
    A single node in the adaptive Huffman tree.

    Internal nodes have left/right children and no symbol.
    Leaf nodes have a symbol and no children.
    The NYT node (Not Yet Transmitted) is a special leaf that
    represents "this symbol has never been seen before".
    """

    def __init__(self, weight, symbol=None, is_nyt=False):
        self.weight   = weight      # frequency count
        self.symbol   = symbol      # None for internal nodes
        self.is_nyt   = is_nyt      # True only for the NYT placeholder
        self.parent   = None
        self.left     = None        # 0-branch
        self.right    = None        # 1-branch
        self.order    = 0           # position in the sibling list (higher = leftmost)

    def is_leaf(self):
        return self.left is None and self.right is None

    def __repr__(self):
        if self.is_nyt:
            return f"Node(NYT, order={self.order})"
        if self.symbol is not None:
            return f"Node(sym={repr(self.symbol)}, w={self.weight}, order={self.order})"
        return f"Node(internal, w={self.weight}, order={self.order})"


# ---------------------------------------------------------------------------
# AdaptiveHuffman — the full encoder/decoder state machine
# ---------------------------------------------------------------------------

class AdaptiveHuffman:
    """
    Implements the FGK adaptive Huffman algorithm.

    One instance represents the shared tree state. The encoder updates
    it as symbols are written; the decoder updates it as symbols are
    read. Because both sides apply the same update rules in the same
    order, they stay in sync without any out-of-band communication.

    Usage (encoding):
        ah = AdaptiveHuffman()
        bits = []
        for char in text:
            bits.extend(ah.encode_symbol(char))
        # bits is a list of 0s and 1s

    Usage (decoding):
        ah = AdaptiveHuffman()
        decoded = ah.decode_bits(bits)
    """

    def __init__(self):
        # The NYT node starts as both the root and the only node.
        # It gets the highest order number so it sorts leftmost.
        self.nyt  = Node(weight=0, is_nyt=True)
        self.root = self.nyt

        # order counter — every new node gets a unique, decreasing order
        # so the sibling property can compare positions numerically
        self.order_counter = 256

        self.nyt.order = self.order_counter

        # symbol → leaf node lookup for O(1) access during encoding
        self.symbol_to_node: dict[str, Node] = {}

    # -----------------------------------------------------------------------
    # Encoding
    # -----------------------------------------------------------------------

    def encode_symbol(self, symbol: str) -> list:
        """
        Encode a single symbol and return its bit sequence.

        If the symbol has been seen before: walk up the tree from its
        leaf to the root, collecting bits (0 for left branch, 1 for right).

        If the symbol is new: emit the NYT path first, then emit the
        symbol's raw 8-bit ASCII representation.

        In both cases, update the tree after emitting.
        """
        bits = []

        if symbol in self.symbol_to_node:
            # Known symbol — encode its current tree path
            node = self.symbol_to_node[symbol]
            bits = self._path_to_root(node)
        else:
            # New symbol — encode NYT path + raw 8-bit character
            bits = self._path_to_root(self.nyt)
            bits += self._symbol_to_bits(symbol)

        # Update tree after encoding (same update the decoder will do)
        self._update(symbol)

        return bits

    def _path_to_root(self, node: Node) -> list:
        """
        Walk from a leaf up to the root, recording which branch we take
        at each step. 0 = left child, 1 = right child.

        We collect bits bottom-up then reverse for top-down order.
        """
        bits = []
        current = node
        while current.parent is not None:
            if current.parent.left is current:
                bits.append(0)
            else:
                bits.append(1)
            current = current.parent
        bits.reverse()
        return bits

    def _symbol_to_bits(self, symbol: str) -> list:
        """
        Convert a character to its 8-bit binary representation.
        We use 8 bits (0-255) to cover the full ASCII range.
        """
        code = ord(symbol)
        return [(code >> (7 - i)) & 1 for i in range(8)]

    # -----------------------------------------------------------------------
    # Decoding
    # -----------------------------------------------------------------------

    def decode_bits(self, bits: list) -> str:
        """
        Decode a flat list of bits back to the original string.

        We walk the tree bit by bit. When we hit a leaf:
          - If it's the NYT node, the next 8 bits are a raw character.
          - Otherwise it's a known symbol.
        After each symbol, update the tree exactly as the encoder did.
        """
        result  = []
        current = self.root
        i       = 0

        while i <= len(bits):
            # Check if current node is a leaf BEFORE consuming the next bit.
            # This handles two cases:
            #   1. The very first symbol when the tree is just the NYT root (no children)
            #   2. All subsequent leaf arrivals after navigation
            if current.is_leaf():
                if current.is_nyt:
                    # NYT leaf: the next 8 bits are a raw ASCII character
                    if i + 8 > len(bits):
                        break   # no more symbols — we're done
                    raw_bits = bits[i:i + 8]
                    i       += 8
                    symbol   = chr(sum(b << (7 - j) for j, b in enumerate(raw_bits)))
                else:
                    symbol = current.symbol

                result.append(symbol)
                self._update(symbol)
                current = self.root   # reset to root for next symbol
                continue

            # Not a leaf — consume the next bit and navigate
            if i >= len(bits):
                break

            bit = bits[i]
            i  += 1

            if bit == 0:
                current = current.left
            else:
                current = current.right

            if current is None:
                # Malformed bitstream — shouldn't happen with valid input
                raise ValueError(f"Unexpected None node at bit position {i}")

        return "".join(result)

    # -----------------------------------------------------------------------
    # Tree update — the heart of FGK
    # -----------------------------------------------------------------------

    def _update(self, symbol: str):
        """
        Update the tree after encoding or decoding a symbol.

        Steps:
          1. If symbol is new: expand the NYT node into a new internal
             node with NYT as left child and new symbol leaf as right child.
          2. Find the highest-order node in the same weight class as the
             node we're about to increment, and swap if needed (sibling property).
          3. Increment weight and move up to parent, repeating until root.
        """
        if symbol not in self.symbol_to_node:
            # First time seeing this symbol — split the NYT node
            self._expand_nyt(symbol)
            # Start incrementing from the new symbol's parent
            node = self.symbol_to_node[symbol].parent
        else:
            node = self.symbol_to_node[symbol]

        # Walk up the tree incrementing weights and maintaining sibling property
        while node is not None:
            # Find the highest-order node with the same weight (sibling property)
            leader = self._find_block_leader(node)

            # Swap with the leader if it's not our parent and not ourselves
            if leader is not node and leader is not node.parent:
                self._swap_nodes(node, leader)

            node.weight += 1
            node = node.parent

    def _expand_nyt(self, symbol: str):
        """
        Replace the NYT leaf with an internal node that has:
          - Left child: new NYT node  (inherits NYT's position)
          - Right child: new symbol leaf
        """
        old_nyt = self.nyt

        # Create the two new children
        self.order_counter -= 1
        new_nyt = Node(weight=0, is_nyt=True)
        new_nyt.order = self.order_counter

        self.order_counter -= 1
        new_leaf = Node(weight=0, symbol=symbol)
        new_leaf.order = self.order_counter

        # Transform old_nyt into an internal node
        old_nyt.is_nyt  = False
        old_nyt.symbol  = None
        old_nyt.left    = new_nyt
        old_nyt.right   = new_leaf

        new_nyt.parent  = old_nyt
        new_leaf.parent = old_nyt

        # Update bookkeeping
        self.nyt = new_nyt
        self.symbol_to_node[symbol] = new_leaf

    def _find_block_leader(self, node: Node) -> Node:
        """
        Find the node with the highest order number among all nodes
        that share the same weight as the given node.

        This is the node we'd swap with to restore the sibling property.
        We search the whole tree with BFS — not the most efficient approach
        but clear and correct for the scale of text we're handling.
        """
        target_weight = node.weight
        leader = node

        # BFS from root to find all nodes with matching weight
        queue = [self.root]
        while queue:
            current = queue.pop(0)
            if current.weight == target_weight and current.order > leader.order:
                leader = current
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)

        return leader

    def _swap_nodes(self, a: Node, b: Node):
        """
        Swap nodes a and b in the tree — meaning swap their positions
        in the tree structure, not just their weights.

        We update the parent pointers to point to each other's position,
        and swap the order numbers so the sibling property accounting
        stays consistent.

        We never swap a node with its own ancestor (that would corrupt
        the tree structure).
        """
        # Make sure neither is an ancestor of the other
        if self._is_ancestor(a, b) or self._is_ancestor(b, a):
            return

        a_parent = a.parent
        b_parent = b.parent

        # Swap positions in parent's child slots
        if a_parent is not None:
            if a_parent.left is a:
                a_parent.left = b
            else:
                a_parent.right = b

        if b_parent is not None:
            if b_parent.left is b:
                b_parent.left = a
            else:
                b_parent.right = a

        # Update parent pointers
        a.parent = b_parent
        b.parent = a_parent

        # Swap order numbers to keep sibling property consistent
        a.order, b.order = b.order, a.order

        # If one of them was the root, update root reference
        if self.root is a:
            self.root = b
        elif self.root is b:
            self.root = a

    def _is_ancestor(self, possible_ancestor: Node, node: Node) -> bool:
        """Walk up from node to see if possible_ancestor appears."""
        current = node.parent
        while current is not None:
            if current is possible_ancestor:
                return True
            current = current.parent
        return False


# ---------------------------------------------------------------------------
# Public API — encode and decode functions
#
# These are the functions called by app.py. They handle the full
# encode→bitstring and bitstring→decode round trip, including
# converting the bit list to/from a compact hex string for transport.
# ---------------------------------------------------------------------------

def encode(text: str) -> tuple[str, int, int]:
    """
    Encode a string using adaptive Huffman and return a hex bitstream.

    Returns:
        compressed_hex  — hex string representation of the compressed bits
        original_bits   — number of bits in uncompressed 8-bit ASCII
        compressed_bits — number of bits in compressed output

    The hex string pads the final byte with zeros if needed, and
    stores the true bit count separately so decoding knows where to stop.
    """
    if not text:
        return "", 0, 0

    ah   = AdaptiveHuffman()
    bits = []

    for char in text:
        bits.extend(ah.encode_symbol(char))

    original_bits   = len(text) * 8
    compressed_bits = len(bits)

    # Pack bits into bytes, padding the last byte if necessary
    compressed_hex = _bits_to_hex(bits)

    return compressed_hex, original_bits, compressed_bits


def decode(compressed_hex: str, compressed_bits: int) -> str:
    """
    Decode a hex bitstream back to the original string.

    Args:
        compressed_hex  — hex string from encode()
        compressed_bits — exact bit count (to strip padding from last byte)
    """
    if not compressed_hex:
        return ""

    bits = _hex_to_bits(compressed_hex)

    # Strip padding bits from the last byte
    bits = bits[:compressed_bits]

    ah = AdaptiveHuffman()
    return ah.decode_bits(bits)


# ---------------------------------------------------------------------------
# Bit ↔ hex helpers
# ---------------------------------------------------------------------------

def _bits_to_hex(bits: list) -> str:
    """Pack a list of 0/1 integers into a hex string, padding to full bytes."""
    # Pad to a multiple of 8
    padded = bits + [0] * ((-len(bits)) % 8)
    result = []
    for i in range(0, len(padded), 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | padded[i + j]
        result.append(byte_val)
    return bytes(result).hex()


def _hex_to_bits(hex_str: str) -> list:
    """Unpack a hex string back to a flat list of 0/1 integers."""
    raw   = bytes.fromhex(hex_str)
    bits  = []
    for byte in raw:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


# ---------------------------------------------------------------------------
# Quick correctness test — run this file directly to verify round-trips
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        "hello",
        "aaaaabbbcc",
        "The quick brown fox",
        "0123456789",
        "aaaa",
        "abcdefghij",
        "mississippi",
        "A" * 50,
        "Hello, World! This is a longer test string with punctuation.",
    ]

    print("Adaptive Huffman FGK — round-trip tests")
    print("-" * 55)

    all_passed = True

    for text in test_cases:
        hex_str, orig_bits, comp_bits = encode(text)
        recovered = decode(hex_str, comp_bits)

        ratio = orig_bits / comp_bits if comp_bits > 0 else 0
        ok    = recovered == text

        if not ok:
            all_passed = False

        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}]  {repr(text[:30]):<35}"
            f"  {orig_bits:>5} → {comp_bits:>5} bits"
            f"  ratio={ratio:.2f}x"
        )
        if not ok:
            print(f"         Expected : {repr(text)}")
            print(f"         Got      : {repr(recovered)}")

    print()
    if all_passed:
        print("  All round-trip tests passed.")
    else:
        print("  SOME TESTS FAILED — check the output above.")
