#include "Game.hpp"
#include "MainMenu.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Window/Event.hpp>


Game::Game() : m_context(std::make_shared<Context>()) {
    // create a window
    m_context->m_window->create(sf::VideoMode(640,360), "Snake Game!", sf::Style::Close);

    // Add first state to m_states
    m_context->m_states->Add(std::make_unique<MainMenu>(m_context));
}

Game::~Game() {

}

void Game::Run() {

    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);

    // For Time
    sf::Clock clock;
    sf::Time timeSinceLastFrame = sf::Time::Zero;

    
    while (m_context->m_window->isOpen()){ 

        timeSinceLastFrame += clock.restart();

        // Make update cycle of our game consistent
        while (timeSinceLastFrame > TIME_PER_FRAME) // Update will happen only if time > 1/60 seconds
        {   
            //
            timeSinceLastFrame -= TIME_PER_FRAME; // decrease the time since last frame by timePerFrame
            

            m_context->m_states->ProcessStateChange(); // State change happens before update cycle begins
            m_context->m_states->GetCurrent()->ProcessInput(); // This will allow the state to handle all the input envents like mouse clicks and key presses
            m_context->m_states->GetCurrent()->Update(TIME_PER_FRAME); // call update method on current state
            m_context->m_states->GetCurrent()->Draw(); // call draw method on current state

       

        }
    }

    
}