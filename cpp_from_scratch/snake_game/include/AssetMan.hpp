#pragma once

#include <map>
#include <memory>
#include <string>

#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Font.hpp>


namespace Engine {

    class AssetMan {

        private:
            std::map<int, std::unique_ptr<sf::Texture>> m_Textures; // to hold textures
            std::map<int, std::unique_ptr<sf::Font>> m_fonts; // to hold fonts

        public:
            AssetMan();
            ~AssetMan();

            // To load new textures and fonts
            void AddTexture(int id, const std::string &filePath, bool wantRepeated = false);
            void AddFont(int id, const std::string &filePath);

            // to get "loaded texture and font" from map [return texture or font by const reference]
            const sf::Texture &GetTexture(int id) const;
            const sf::Font &GetFont(int id) const;

    };

}
