#include "GameOver.hpp"
#include "GamePlay.hpp"

#include <SFML/Window/Event.hpp>

GameOver::GameOver(std::shared_ptr<Context>& context) 
        : m_context(context),
        m_isRetryButtonSelected(true), m_isRetryButtonPressed(false),
        m_isExitButtonSelected(false), m_isExitButtonPressed(false)
{

}
GameOver::~GameOver() {

}
void GameOver::Init() {
    m_context->m_assets->AddFont(MAIN_FONT, "assets/fonts/Pacifico-Regular.ttf"); // Load Font

    // Initialize game Title
    m_gameOverTitle.setFont(m_context->m_assets->GetFont(MAIN_FONT)); // get the loaded font and set it to gameTitle
    m_gameOverTitle.setString("Game Over"); // add title   
    m_gameOverTitle.setOrigin(m_gameOverTitle.getLocalBounds().width / 2,
                            m_gameOverTitle.getLocalBounds().height / 2);
    m_gameOverTitle.setPosition(m_context->m_window->getSize().x/2,
                            m_context->m_window->getSize().y/2 - 150.f); 

    // Initialize Play button
    m_retryButton.setFont(m_context->m_assets->GetFont(MAIN_FONT)); // get the loaded font and set it to gameTitle
    m_retryButton.setString("Retry"); // add title
    m_retryButton.setOrigin(m_retryButton.getLocalBounds().width / 2,
                            m_retryButton.getLocalBounds().height / 2);
    m_retryButton.setPosition(m_context->m_window->getSize().x/2,
                            m_context->m_window->getSize().y/2 - 25.f);
    m_retryButton.setCharacterSize(20);

    // Initialize Exit button
    m_exitButton.setFont(m_context->m_assets->GetFont(MAIN_FONT)); // get the loaded font and set it to gameTitle
    m_exitButton.setString("Exit"); // add title
    m_exitButton.setOrigin(m_exitButton.getLocalBounds().width / 2,
                            m_exitButton.getLocalBounds().height / 2);
    m_exitButton.setPosition(m_context->m_window->getSize().x/2,
                            m_context->m_window->getSize().y/2 + 25.f);
    m_exitButton.setCharacterSize(20);
}
void GameOver::ProcessInput() {
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
                case sf::Keyboard::Up:{

                    if (!m_isRetryButtonSelected) {
                        m_isRetryButtonSelected = true;
                        m_isExitButtonSelected = false;
                    }
                    break;
                }
                case sf::Keyboard::Down:{
                    if (!m_isExitButtonSelected) {
                        m_isRetryButtonSelected = false;
                        m_isExitButtonSelected = true;
                    }
                    break;
                }
                case sf::Keyboard::Return:{ // Detect key press of Enter key
                    m_isRetryButtonPressed = false;
                    m_isExitButtonPressed = false;
                    if (m_isRetryButtonSelected) {
                        m_isRetryButtonPressed = true;
                    } else {
                        m_isExitButtonPressed = true;
                    }
                    break;
                }
                default:{
                    break;
                }
                    
            }
        }        
    }
}

void GameOver::Update(sf::Time deltaTime) {


    if (m_isRetryButtonSelected) {
        m_retryButton.setFillColor(sf::Color::Green);
        m_exitButton.setFillColor(sf::Color::White);

    }else {
        m_exitButton.setFillColor(sf::Color::Red);
        m_retryButton.setFillColor(sf::Color::White);

    }

    if (m_isRetryButtonPressed){
       
        // Go to GamePlay state by replacing GameOver state
        m_context->m_states->Add(std::make_unique<GamePlay>(m_context), true);

    } else if(m_isExitButtonPressed){
        m_context->m_window->close();
    }


}
void GameOver::Draw() {
    
    m_context->m_window->clear();
    m_context->m_window->draw(m_gameOverTitle);
    m_context->m_window->draw(m_retryButton);
    m_context->m_window->draw(m_exitButton);
    m_context->m_window->display(); 
}