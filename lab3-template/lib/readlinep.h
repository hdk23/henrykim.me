/* 
 * readlinep - a utility function to safely read one line of input,
 * of any length, and return a pointer to a null-terminated string.
 *
 * David Kotz, April 2017, 2019
 */

#ifndef __READLINEP_H__
#define __READLINEP_H__

#include <stdio.h>

/* readlinep, freadlinep:
 *
 * Read a line from the file into a null-terminated string.
 *   readlinep() is equivalent to freadlinep(stdin).
 * We assume:
 *   caller provides (for freadlinep) a valid FILE open for reading.
 * We return:
 *   a pointer to a malloc'd string; caller must later free() the pointer;
 *   a NULL if EOF or any error.
 * We guarantee:
 *   null-terminated string in the buffer;
 *   no newline in the buffer.
 * Caller is responsible for:
 *   eventually calling free() on the pointer we return.
 * Notes:
 *   If end-of-file is reached before newline, that is not an error.
 *   The string returned contains all characters before that end-of-file.
 */
extern char *freadlinep(FILE *fp);
static inline char *readlinep(void) { return freadlinep(stdin); }

#endif //  __READLINEP_H__
