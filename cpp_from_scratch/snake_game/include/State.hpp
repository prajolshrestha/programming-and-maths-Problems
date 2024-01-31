#pragma once

#include <SFML/System/Time.hpp>

namespace Engine {
    class State { // State is a base class
        public:
            State(){};
            virtual ~State(){}; // always mark destructors of base class as virtual (o/w destruction of derrived class object will not happen properly)

            ///// These pure virtual function should be defined in derived class. (These four method is necessary for every state.)
            virtual void Init() = 0;  // All initial setup will happen in this method.
            virtual void ProcessInput() = 0;// To handle key press and mouse press
            virtual void Update(sf::Time deltaTime) = 0; // Reacts to the inputs handeled in processInput state (by making changes in the state like recalculate position of text)
                                        //deltaTime is elapsed time since last update. This ensures same FPS on every machine.
                                        // Explanation: Update method gets called multiple times inside an infinite while loop.
                                                        // and depending upon how fast or slow your machine is, the update done per second will vary.
                                                        // This results game to run faster/slower depending on machine. 
            virtual void Draw() = 0; // To draw all the sprites and text of the state on the rendering window.



            //// To pause and start a state(only necessary for some state.) // Optional: Blank implementation.
            virtual void Pause(){}; // When we pause a game, Game world will remain visible to us. But a transparent layer (pause menu) will be placed on of it. & It blocks any input to the game world.
            virtual void Start(){};
    
    };


} // namespace Engine

