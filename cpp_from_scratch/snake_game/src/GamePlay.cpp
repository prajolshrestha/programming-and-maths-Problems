#include "GamePlay.hpp"

#include <SFML/Window/Event.hpp>

#include <stdlib.h>
#include <time.h>

GamePlay::GamePlay(std::shared_ptr<Context>& context) : 
m_context(context), m_snakeDirection({16.f,0.f}), m_elapsedTime(sf::Time::Zero){

    srand(time(nullptr)); // random number generator
}

GamePlay::~GamePlay(){

}

void GamePlay::Init() {
    // Load all textures
    m_context->m_assets->AddTexture(GRASS, "assets/textures/grass.png", true);
    m_context->m_assets->AddTexture(FOOD, "assets/textures/food.png");
    m_context->m_assets->AddTexture(WALL, "assets/textures/wall.png", true);
    m_context->m_assets->AddTexture(SNAKE, "assets/textures/snake.png");

    // Review:
    // setTextureRect() : what to display
    // setPosition() : where to display

    // Set grass textures on the sprites
    m_grass.setTexture(m_context->m_assets->GetTexture(GRASS));
    m_grass.setTextureRect(m_context->m_window->getViewport(m_context->m_window->getDefaultView()));

    // set wall textures on the sprites
    for (auto& wall : m_walls){ // auto& to get ref to each sprite
        wall.setTexture(m_context->m_assets->GetTexture(WALL));
    }
    m_walls[0].setTextureRect({0,0,m_context->m_window->getSize().x,16});
    m_walls[1].setTextureRect({0,0,m_context->m_window->getSize().x,16});
    m_walls[1].setPosition(0, m_context->m_window->getSize().y -16); //set position

    m_walls[2].setTextureRect({0,0,16,m_context->m_window->getSize().y});
    m_walls[3].setTextureRect({0,0,16,m_context->m_window->getSize().y});
    m_walls[3].setPosition(m_context->m_window->getSize().x-16 ,0);// set position

    // set food testures on the sprites
    m_food.setTexture(m_context->m_assets->GetTexture(FOOD));
    m_food.setPosition(m_context->m_window->getSize().x/2, m_context->m_window->getSize().y/2);


    // Initialize snake
    m_snake.Init(m_context->m_assets->GetTexture(SNAKE));

};
void GamePlay::ProcessInput() {
    // Window update
    sf::Event event;
    while (m_context->m_window->pollEvent(event))
    {
        if (event.type == sf::Event::Closed)
        {
             m_context->m_window->close();
        }
        // change direction according to input
        else if (event.type == sf::Event::KeyPressed){
            sf::Vector2f newDirection = m_snakeDirection; // to control direction
            switch (event.key.code)
            {
            case sf::Keyboard::Up:
                newDirection = {0.f, -16.f};
                break;
            case sf::Keyboard::Down:
                newDirection = {0.f, 16.f};
                break;
            case sf::Keyboard::Left:
                newDirection = {-16.f, 0.f};
                break;
            case sf::Keyboard::Right:
                newDirection = {16.f, 0.f};
                break;
            
            default:
                break;
            }
            // change direction
            if (std::abs(m_snakeDirection.x) != std::abs(newDirection.x) ||
                std::abs(m_snakeDirection.y) != std::abs(newDirection.y)){ // to stop 180degree rotation
                 m_snakeDirection = newDirection; // direction changed only when key(other than 180 degree) is pressed
                }
        }
    }
};
void GamePlay::Update(sf::Time deltaTime) {
    m_elapsedTime += deltaTime; // controls the speed of snake

    if (m_elapsedTime.asSeconds() > 0.1){

        bool isOnWall = false;

        for (auto& wall : m_walls){
            if (m_snake.IsOn(wall)){  // if collision with wall
                //TODO
                // Game over
                break;
            }
        }
        
        if (m_snake.IsOn(m_food)){ // if collision with food
            m_snake.Grow(m_snakeDirection);

            int x = 0, y = 0; // new coordinates for food sprites

            // we need the loc of food within the limits of render window
            //x = rand() % m_context->m_window->getSize().x;
            //y = rand() % m_context->m_window->getSize().y;

            // we want loc of food inside of wall , not on wall
            x = std::clamp<int>(rand() % m_context->m_window->getSize().x, 16, m_context->m_window->getSize().x - 2*16);
            y = std::clamp<int>(rand() % m_context->m_window->getSize().y, 16, m_context->m_window->getSize().y - 2*16);


            m_food.setPosition(x,y);
        }
        else {
            m_snake.Move(m_snakeDirection); // move snake
        }


        m_elapsedTime = sf::Time::Zero;
    }
    
};
void GamePlay::Draw() {

    m_context->m_window->clear();
    m_context->m_window->draw(m_grass); // lets draw grass
    for (auto& wall : m_walls){
        m_context->m_window->draw(wall); // Draw 4 walls
    }
    m_context->m_window->draw(m_food); // Draw food
    m_context->m_window->draw(m_snake); // Draw snake
    m_context->m_window->display();
};
void GamePlay::Pause() {

};
void GamePlay::Start() {

};