class BitboardEnv:
    def __init__(self):
        self.shifts = [8, -8, 1, -1, 9, 7, -7, -9]
        self.masks = [
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFEFEFEFEFEFEFEFE, 0x7F7F7F7F7F7F7F7F,
            0xFEFEFEFEFEFEFEFE, 0x7F7F7F7F7F7F7F7F,
            0xFEFEFEFEFEFEFEFE,
            0x7F7F7F7F7F7F7F7F 
        ]

    def matrix_to_bitboard(self, board):
        white_bits, black_bits = 0, 0
        for i in range(64):
            val = board[i // 8, i % 8]
            if val == 1: white_bits |= (1 << i)
            elif val == -1: black_bits |= (1 << i)
        return white_bits, black_bits

    def get_legal_moves(self, p_bits, o_bits):
        empty = ~(p_bits | o_bits) & 0xFFFFFFFFFFFFFFFF
        legal_moves = 0
        for s, m in zip(self.shifts, self.masks):
            targets = self._shift(p_bits, s, m) & o_bits
            for _ in range(5):
                targets |= self._shift(targets, s, m) & o_bits
            legal_moves |= self._shift(targets, s, m) & empty
        return legal_moves

    def apply_move(self, p_bits, o_bits, move_bit):
        flips = 0
        for s, m in zip(self.shifts, self.masks):
            candidate_flips = 0
            curr = self._shift(move_bit, s, m)
            while curr & o_bits:
                candidate_flips |= curr
                curr = self._shift(curr, s, m)
            if curr & p_bits:
                flips |= candidate_flips
        return (p_bits | move_bit | flips), (o_bits & ~flips)

    def _shift(self, bits, s, m):
        if s > 0: return (bits << s) & m
        else: return (bits >> abs(s)) & m