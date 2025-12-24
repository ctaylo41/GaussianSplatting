//
//  main.cpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-24.
//


#include <iostream>
#include <Metal/Metal.hpp>
#include "mtl_engine.hpp"

int main() {
    MTLEngine engine;
    engine.init();
    engine.run();
    engine.cleanup();
    return 0;
}
