/*
 * pyse_hooks.h -- hooks that the Surface Evolver sources call when compiled
 * with -DPYSE.  Each patched call site in src/ is wrapped in #ifdef PYSE, so
 * the stock Makefile build is unaffected.
 *
 * In "library mode" (pyse_library_mode != 0) these hooks
 *   - capture text written to stdout/stderr through outstring()/erroutstring(),
 *   - turn my_exit() into a longjmp back to the Python binding,
 *   - stop Evolver from reading stdin,
 *   - record every kb_error() so it can be raised as a Python exception.
 * In CLI mode (pyse_library_mode == 0) they do nothing and Evolver behaves
 * exactly like the standalone executable.
 */
#ifndef PYSE_HOOKS_H
#define PYSE_HOOKS_H

extern int pyse_library_mode;

/* Called by my_exit(); never returns in library mode. */
void pyse_exit(int code);

/* Return nonzero if the text was consumed (library mode). */
int pyse_outstring(const char *text);
int pyse_erroutstring(const char *text);

/* Called for every kb_error() with its raw message and mode. */
void pyse_note_error(int errnum, const char *emsg, int mode);

/* Called when Evolver wants a line from stdin.
   Returns -1: not handled (read stdin as usual), 0: EOF, 1: line in buf. */
int pyse_read_stdin(const char *prompt, char *buf, int max);

#endif
