#pragma once

#include <memory>
#include <array>

#include <SFML/Graphics/Sprite.hpp>

#include "Game.hpp"
#include "State.hpp"
#include "Snake.hpp"

class GamePlay : public Engine::State {

    private:
        std::shared_ptr<Context> m_context;

        // Lets add grass, food and wall
        sf::Sprite m_grass;
        sf::Sprite m_food;
        std::array<sf::Sprite, 4> m_walls;

        // Lets add snake : we need list of multiple sprite object to represent snake
        Snake m_snake;

        // direction
        sf::Vector2f m_snakeDirection;

        // Timer
        sf::Time m_elapsedTime;

    public:
        GamePlay(std::shared_ptr<Context> &context);
        ~GamePlay();

        void Init() override;
        void ProcessInput() override;
        void Update(sf::Time deltaTime) override;
        void Draw() override;
        void Pause() override;
        void Start() override;
};