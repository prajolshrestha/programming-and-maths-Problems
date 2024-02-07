#include "PauseGame.hpp"

#include <SFML/Window/Event.hpp>

PauseGame::PauseGame(std::shared_ptr<Context>& context) : m_context(context){

}

PauseGame::~PauseGame() {

}

void PauseGame::Init() {
    //m_context->m_assets->AddFont(MAIN_FONT, "assets/fonts/Pacifico-Regular.ttf"); // Load Font
    // Please do not load asset again ! it leads to segmentation fault! ie pointing to object that has already been destroyed!
    // Initialize pause Title
    m_pauseTitle.setFont(m_context->m_assets->GetFont(MAIN_FONT)); // get the loaded font and set it to gameTitle
    m_pauseTitle.setString("Paused"); // add title   
    m_pauseTitle.setOrigin(m_pauseTitle.getLocalBounds().width / 2,
                            m_pauseTitle.getLocalBounds().height / 2);
    m_pauseTitle.setPosition(m_context->m_window->getSize().x/2,
                            m_context->m_window->getSize().y/2); 

    
}
void PauseGame::ProcessInput() {
    // Window update
    sf::Event event;
    while (m_context->m_window->pollEvent(event))
    {
        if (event.type == sf::Event::Closed)
        {
             m_context->m_window->close();
        }

        else if (event.type == sf::Event::KeyPressed){
            // check which key was pressed
            switch (event.key.code) // It stores the key that was pressed
            {
                case sf::Keyboard::Escape:{

                    m_context->m_states->PopCurrent();
                    break;
                }
                default:{
                    break;
                }
                    
            }
        }        
    }
}

void PauseGame::Update(sf::Time deltaTime) {



}
void PauseGame::Draw() {

    
    m_context->m_window->draw(m_pauseTitle);
    
    m_context->m_window->display(); 
}