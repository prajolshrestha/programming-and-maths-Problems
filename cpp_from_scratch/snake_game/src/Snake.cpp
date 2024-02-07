#include "Snake.hpp"

Snake::Snake() : m_body(std::list<sf::Sprite>(4)) {
    m_head = --m_body.end();
    m_tail = m_body.begin();
}

Snake::~Snake() {

}

void Snake::Init(const sf::Texture& texture){
    float x = 16.f;
    // lets set the texture for all m_body with input texture // 4 piece of snake
    for(auto& piece : m_body){  //get each element as a ref
        piece.setTexture(texture); 
        piece.setPosition({x, 16.f}); // set initial position of snake body 
        x += 16.f; // later we inc x by 16
    }

} 
void Snake::Move(const sf::Vector2f& direction){
    // It is called after fixed interval of time to keep the snake moving
    // we will not move whole snake, but we take tail sprite and change its position such that it is placed right next to the head

    m_tail->setPosition(m_head->getPosition() + direction); // tail moved infront of head
    m_head = m_tail; // now old tail is our new head
    ++m_tail;       // new tail is second last piece of snake

    // when m_tail reach the end of m_body
    if (m_tail == m_body.end()){
        m_tail = m_body.begin(); // reset m_tail
    }
}
bool Snake::IsOn(const sf::Sprite& other) const{
    // collision detection // just detect if the head of snake intersects other sprite
    return other.getGlobalBounds().intersects(m_head->getGlobalBounds());
}
void Snake::Grow(const sf::Vector2f& direction){
    // Grow body of snake 
    
    // create new piece for snake
    sf::Sprite newPiece;
    newPiece.setTexture(*(m_body.begin()->getTexture()));
    newPiece.setPosition(m_head->getPosition() + direction);

    // add new piece to snake body
    m_head = m_body.insert(++m_head, newPiece);

}
bool Snake::IsSelfIntersecting() const {

    bool flag = false;
    // Snake intersecting itself??
    for (auto piece = m_body.begin(); piece != m_body.end(); ++piece){
        if (m_head != piece){
            flag = IsOn(*piece);

            if (flag){
                break;
            }
        }
    }
    return flag;
}
void Snake::draw(sf::RenderTarget& target, sf::RenderStates states) const{
    for (auto& piece: m_body){
        target.draw(piece); // pass every element to draw method of target
    }
}