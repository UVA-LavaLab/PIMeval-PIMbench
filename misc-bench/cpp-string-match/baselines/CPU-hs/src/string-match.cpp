#include <hs.h>
#include <iostream>

int main() {
    hs_database_t* database;
    hs_compile_error_t* compileErr;
    const char* pattern = "example";

    // Compile the pattern
    if (hs_compile(pattern, HS_FLAG_CASELESS, HS_MODE_BLOCK, nullptr, &database, &compileErr) != HS_SUCCESS) {
        std::cerr << "Error compiling pattern: " << compileErr->message << std::endl;
        hs_free_compile_error(compileErr);
        return 1;
    }

    std::cout << "Pattern compiled successfully!" << std::endl;

    hs_free_database(database);
    return 0;
}
