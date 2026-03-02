# ==============================================================================
# OpenImpala - Unit Tests (Catch2)
#
# These tests exercise pure C++ logic (config parsing, computation helpers)
# without requiring MPI, AMReX, or HYPRE. They compile fast and run in
# milliseconds, making them ideal for rapid iteration.
# ==============================================================================

include(Catch)

# --- PhysicsConfig unit tests ---
add_executable(unit_PhysicsConfig
    "${CMAKE_CURRENT_SOURCE_DIR}/test_PhysicsConfig.cpp"
)
target_include_directories(unit_PhysicsConfig PRIVATE
    ${CMAKE_SOURCE_DIR}/src/props
)
target_compile_definitions(unit_PhysicsConfig PRIVATE OPENIMPALA_UNIT_TEST)
target_link_libraries(unit_PhysicsConfig PRIVATE Catch2::Catch2WithMain)

catch_discover_tests(unit_PhysicsConfig)

# --- ResultsJSON unit tests ---
add_executable(unit_ResultsJSON
    "${CMAKE_CURRENT_SOURCE_DIR}/test_ResultsJSON.cpp"
)
target_include_directories(unit_ResultsJSON PRIVATE
    ${CMAKE_SOURCE_DIR}/src/props
)
target_compile_definitions(unit_ResultsJSON PRIVATE OPENIMPALA_UNIT_TEST)
target_link_libraries(unit_ResultsJSON PRIVATE
    Catch2::Catch2WithMain
    nlohmann_json::nlohmann_json
)

catch_discover_tests(unit_ResultsJSON)
