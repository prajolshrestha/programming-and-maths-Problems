#pragma once

#include <list> // If we use vector, when vector is dynamically increased, it relocates.
            // list does not relocate.

#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Drawable.hpp> // to draw snake
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RenderStates.hpp>


class Snake : public sf::Drawable{

private:
    // Represent body, head and tail of a snake
    std::list<sf::Sprite> m_body;
    
    std::list<sf::Sprite>::iterator m_head; // used in move and grow method
    std::list<sf::Sprite>::iterator m_tail; // used in move and grow method

public:
    Snake();
    ~Snake();

    void Init(const sf::Texture& texture);  // Initialize snake
    void Move(const sf::Vector2f& direction); //move the snake in given direction
    bool IsOn(const sf::Sprite& other) const; // snake collided with other object? eg. food or wall
    void Grow(const sf::Vector2f& direction); // Grow snake (at head in given direction)
    bool IsSelfIntersecting() const;

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
};