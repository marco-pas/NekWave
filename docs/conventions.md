# NekWave Naming Conventions

## 1. Files
* **Extensions:** C++ source files have a `.cpp` extension and use `.hpp` for headers. C source files have a `.c` extension and use `.h` for headers.
* **Corresponding Headers:** For source file `file.c`/`file.cpp`, declarations that are visible outside the source file should go into a correspondingly named header: `file.h` or `file.hpp`. Some code may deviate from this rule to improve readability and/or usability of the API, but this should then be clearly documented.
* **Implementation Headers:** There can also be a `file-impl.h` or `file-impl.hpp` file that declares classes or functions that are not accessible outside the module. If the whole file only declares symbols internal to the module, then the `-impl` suffix is omitted.
* **Local Declarations:** In most cases, declarations that are not used outside a single source file are in the source file.
* **Documentation:** Use suffix `-doc.h` or `-doc.hpp` for files that contain only Doxygen documentation for some module or such.
* **C++ Class Matching:** For C++ files, prefer naming the file the same as the (main) class it contains. Currently all file names are all-lowercase, even though class names contain capital letters. It is OK to use commonly known abbreviations, and/or omit the name of the containing directory if that would cause unnecessary repetition.
* **Uniqueness:** Avoid having multiple files with the same name in different places within the same library. In addition to making things harder to find, C++ source files with the same name can cause obscure problems with some compilers.

## 2. Common Guidelines for C and C++ Code
* **Macros:** Preprocessor macros should be all upper-case. Do not use leading underscores, as all such names are reserved according to the C/C++ standard.
* **Include Guards:** Name include guards like `NW_DIRNAME_HEADERNAME_H` or `NW_DIRNAME_HEADERNAME_HPP`.
* **Boolean Variables:** Boolean variables are always named with a `b` prefix, followed by a CamelCase name (e.g., `bIsValid`).
* **Enums:** Enum values are named with an `e` prefix. For enum types exposed widely in the codebase, this is followed typically by a part that makes the enum values not conflict with other enums in the same scope. 
  * In C code, this is typically an all-lowercase acronym (e.g., `epbcNONE`).
  * In C++, the same approach may be used, or the name of the enum type is used (e.g., `eHelpOutputFormat_Console`).
* **Abbreviations:** Avoid abbreviations that are not obvious to a general reader.
* **Acronyms:** If you use acronyms (e.g., PME, DD) in names, follow the Microsoft policy on casing: two letters is uppercase (`DD`), three or more is lowercase (`Pme`).

## 3. C Code
* **General:** All function and variable names are lowercase, with underscores as word separators where needed for clarity.
* **API Functions:** All functions that are part of the public API should start with `nw_`. Preferably, other functions should as well. Some parts of the code use a `_nw_` prefix for internal functions, but strictly speaking, these are reserved names, so a trailing underscore would be better.

## 4. C++ Code
* **General Casing:** Use CamelCase for all names. Start types (such as classes, structs, and typedefs) with a capital letter, other names (functions, variables) with a lowercase letter. You may use an all-lowercase name with underscores if your class closely resembles an external construct named that way.
* **Interfaces & Base Classes:** C++ interfaces are named with an `Interface` suffix, and abstract base classes with an `Abstract` prefix.
* **Member Variables:** Member variables are named with a trailing underscore (e.g., `value_`).
* **Accessors:** Accessors for a variable `foo_` are named `foo()` and `setFoo()`.
* **Global/Static Variables:** Global variables are named with a `g_` prefix. Static class variables are named with an `s_` prefix.
* **Constants:** Global constants are often named with a `c_` prefix.
* **Class/File Alignment:** If the main responsibility of a file is to implement a particular class, then the name of the file should match that class, except for possible abbreviations to avoid repetition in file names.

## 5. Unit Tests
* **Test Fixtures:** Test fixtures (the first parameter to `TEST`/`TEST_F`) are named with a `Test` suffix.
* **Base Classes:** Classes meant as base classes for test fixtures are named with a `TestBase` or `Fixture` suffix.
* **CTest:** The CTest test is named with CamelCase, ending with `Tests` (e.g., `OptionsUnitTests`).
* **Binaries:** The test binary is named with the name of the module and a `-test` suffix.
