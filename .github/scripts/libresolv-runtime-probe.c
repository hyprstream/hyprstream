/* Intentionally empty: CI links this binary with --no-as-needed -lresolv so
 * the runtime loader must resolve libresolv.so.2 before main is entered. */
int main(void) {
    return 0;
}
