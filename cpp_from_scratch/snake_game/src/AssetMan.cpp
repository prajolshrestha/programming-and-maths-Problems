#include "AssetMan.hpp"

Engine::AssetMan::AssetMan() {

}

Engine::AssetMan::~AssetMan() {

}

/// Add Texture
void Engine::AssetMan::AddTexture(int id, const std::string &filePath, bool wantRepeated){

    auto texture = std::make_unique<sf::Texture>(); //unique_ptr that points to instance of sf::Texture . // texture is a pointer to an object

                // std::make_unique:
                // It creates a dynamically allocated object and returns a unique pointer owning that object.
                // It creates a sf::Texture object on the heap and assigns the ownership of the object to the "texture" unique pointer

    if (texture->loadFromFile(filePath)){ // True if loading is successful [-> access members of a class through a pointer]

        texture->setRepeated(wantRepeated);
        m_Textures[id] = std::move(texture); // store the texture to m_textures

    }
}

// Add Fonts
void Engine::AssetMan::AddFont(int id, const std::string &filePath){

    auto font = std::make_unique<sf::Font>(); // unique_ptr to sf::Font object

    if (font->loadFromFile(filePath)){

        m_fonts[id] = std::move(font);
    }
}
          
const sf::Texture &Engine::AssetMan::GetTexture(int id) const{

    return *(m_Textures.at(id).get()); // get texture(unique_ptr) by id and use get() to get raw pointer & Finally derefrence it by *
                            // Dereference: to return a reference to the underlying texture object
                            // this gives access to actual object
}

const sf::Font &Engine::AssetMan::GetFont(int id) const{

    return *(m_fonts.at(id).get());
}