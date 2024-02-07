#pragma once

#include <memory>

#include <SFML/Graphics/RenderWindow.hpp>

#include <AssetMan.hpp>
#include <StateMan.hpp>

enum AssetID {
    MAIN_FONT = 0,
    GRASS,
    FOOD,
    WALL,
    SNAKE
};

struct Context{ // It holds asset_manager, state_manager & render_window
    
    // This structure holds unique_ptr to all members // Define pointers
    std::unique_ptr<Engine::AssetMan> m_assets;
    std::unique_ptr<Engine::StateMan> m_states;
    std::unique_ptr<sf::RenderWindow> m_window;

    // Default constructor
    Context(){
        // Iniitialize all unique pointers using std::make_unique
        m_assets = std::make_unique<Engine::AssetMan>();
        m_states = std::make_unique<Engine::StateMan>();
        m_window = std::make_unique<sf::RenderWindow>();
    }

};
// Basically, we pass an object of this structure to each of the states,
// so that, the state can access the assets, load new states and draw the rendering window


class Game {

private:
    std::shared_ptr<Context> m_context; // shared_ptr because context will be common for all the states
    const sf::Time TIME_PER_FRAME = sf::seconds(1.f/60.f); // Constant time object (60 frames per seconds)

public:
    Game();
    ~Game();

    void Run();
};