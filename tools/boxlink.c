/* boxlink: do a box diagram's connectors meet? Reads UTF-8 lines (from a file), one code point per column.
 * Checks: every cell that connects down sits above one that connects up and vice versa; a down arrow lands on a
 * horizontal edge; a left/right arrow touches a vertical edge; every line has the same width. Prints each fault
 * with its line and column, then a count; exit status = faults. */
#include <stdio.h>
#include <string.h>
enum { U = 1, D = 2, L = 4, R = 8, MAXL = 200, MAXC = 300 };
static unsigned cp[MAXL][MAXC];
static int nc[MAXL];
static int conn(unsigned c)
{
	switch (c) {
	case 0x2502: case 0x2551: return U | D;          /* │ ║ */
	case 0x2500: case 0x2550: return L | R;          /* ─ ═ */
	case 0x250C: case 0x2554: return D | R;          /* ┌ ╔ */
	case 0x2510: case 0x2557: return D | L;          /* ┐ ╗ */
	case 0x2514: case 0x255A: return U | R;          /* └ ╚ */
	case 0x2518: case 0x255D: return U | L;          /* ┘ ╝ */
	case 0x251C: return U | D | R;                   /* ├ */
	case 0x2524: return U | D | L;                   /* ┤ */
	case 0x252C: return D | L | R;                   /* ┬ */
	case 0x2534: return U | L | R;                   /* ┴ */
	case 0x253C: return U | D | L | R;               /* ┼ */
	case 0x25BC: return U;                           /* ▼ fed from above */
	case 0x25B2: return D;                           /* ▲ */
	case 0x25B6: return L;                           /* ▶ fed from the left */
	case 0x25C0: return R;                           /* ◀ */
	default: return 0;
	}
}
static unsigned at(int l, int c, int nl) { return l < 0 || l >= nl || c < 0 || c >= nc[l] ? ' ' : cp[l][c]; }
int main(int argc, char **argv)
{
	FILE *f = argc > 1 ? fopen(argv[1], "r") : NULL;
	if (!f) { printf("usage: boxlink FILE\n"); return 99; }
	char line[4096];
	int nl = 0, faults = 0;
	while (nl < MAXL && fgets(line, sizeof line, f)) {
		const unsigned char *s = (const unsigned char *)line;
		int n = 0;
		while (*s && *s != '\n' && n < MAXC) {
			unsigned c = *s++;
			if (c >= 0xF0) { c = (c & 7) << 18; c |= (*s++ & 63u) << 12; c |= (*s++ & 63u) << 6; c |= *s++ & 63u; }
			else if (c >= 0xE0) { c = (c & 15) << 12; c |= (*s++ & 63u) << 6; c |= *s++ & 63u; }
			else if (c >= 0xC0) { c = (c & 31) << 6; c |= *s++ & 63u; }
			cp[nl][n++] = c;
		}
		nc[nl++] = n;
	}
	fclose(f);
	for (int l = 1; l < nl; l++) if (nc[l] != nc[0]) { printf("line %d: width %d, line 1 has %d\n", l + 1, nc[l], nc[0]); faults++; }
	for (int l = 0; l < nl; l++)
		for (int c = 0; c < nc[l]; c++) {
			const unsigned ch = cp[l][c];
			const int k = conn(ch), below = conn(at(l + 1, c, nl)), above = conn(at(l - 1, c, nl));
			if (ch == 0x25BC) { if (!(below & (L | R))) { printf("line %d col %d: arrow lands on no horizontal edge\n", l + 1, c + 1); faults++; } }
			else if ((k & D) && !(below & U)) { printf("line %d col %d: connects down, nothing below\n", l + 1, c + 1); faults++; }
			if ((k & U) && !(above & D)) { printf("line %d col %d: connects up, nothing above\n", l + 1, c + 1); faults++; }
			if (ch == 0x25C0 && !(conn(at(l, c - 1, nl)) & (U | D))) { printf("line %d col %d: left arrow touches no vertical edge\n", l + 1, c + 1); faults++; }
			if (ch == 0x25B6 && !(conn(at(l, c + 1, nl)) & (U | D))) { printf("line %d col %d: right arrow touches no vertical edge\n", l + 1, c + 1); faults++; }
		}
	printf("boxlink: %d lines, width %d, %d faults\n", nl, nl ? nc[0] : 0, faults);
	return faults;
}
