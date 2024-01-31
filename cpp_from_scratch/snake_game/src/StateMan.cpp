#include "StateMan.hpp" // Solve the issue with "includepath": ${workspaceFolder}/include    [ctrl + .]
                        // It should look for headers under include folder.

//Constructor
Engine::StateMan::StateMan() : m_add(false), m_replace(false), m_remove(false) { // at construction, all boolean member is initialized with false

}

// Destructor
Engine::StateMan::~StateMan() {

}




void Engine::StateMan::Add (std::unique_ptr<State> toAdd, bool replace){

    m_add = true; // update add

    // When state is being added
    m_newState = std::move(toAdd); //As these are unique_ptr we will have to use STD move to transfer the ownership from "toAdd" to "m_newState"

    m_replace = replace; // update replace

}

///
void Engine::StateMan::PopCurrent(){

    m_remove = true;
}

// Actual modification of state stack
void Engine::StateMan::ProcessStateChange(){

    // Process remove [Remove current state(top of stack) & start a state on top of stack]
    if (m_remove && (!m_stateStack.empty())){ // if remove and state stack is not empty

        m_stateStack.pop(); // remove a state

        if (!m_stateStack.empty()){ // if stack is not empty

            m_stateStack.top()->Start(); // start any state that is on top of stack
        }

        m_remove = false; // reset remove [if not done it keep on remove all state on the stack]
    }

    // Process add [Pause current state(top of stack) & add new state & initialize & start]
    if (m_add){
        // remove a state
        if (m_replace && (!m_stateStack.empty())){

            m_stateStack.pop(); 
            m_replace = false; // reset
        }

        // pause current state
        if (!m_stateStack.empty()){

            m_stateStack.top()->Pause(); 
        }

        // add new state on the stack
        m_stateStack.push(std::move(m_newState));
        m_stateStack.top()->Init();
        m_stateStack.top()->Start();
        m_add = false; // reset
    }

}


std::unique_ptr<Engine::State>& Engine::StateMan::GetCurrent(){

    // Return state that is at the top of the stack
    return m_stateStack.top(); 
}