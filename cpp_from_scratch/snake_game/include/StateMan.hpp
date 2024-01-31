#pragma once

#include <stack>
#include <memory>

#include <State.hpp>

namespace Engine{
    class StateMan { // State Manager

    private:
        // Stack to store the states (ie unique_ptr to state obj)
        std::stack<std::unique_ptr<State>> m_stateStack; // Our stack "m_stateStack" will store unique_ptr to State object.

        // To store the new states  // [Caution: State stack should only be modified at the start of next update cycle.]
        std::unique_ptr<State> m_newState; // Required because we don't want to push new state on the stack while other state is executing its update method.
                                        // Instead, we store the new state in "m_newState" member & add it to stack only after the update cycle of current state completes.

        //Boolean members [modified by Add and popCurrent methods] 
        //==> Depending on these boolean values the ProcessStateChange method will make modification to the state Stack
        bool m_add;
        bool m_replace;
        bool m_remove;

    public:
        StateMan();
        ~StateMan();

        //// Add and PopCurrent method will just store the intent that we want to add or remove a state.
        void Add (std::unique_ptr<State> toAdd, bool replace = false);// toAdd = unique_ptr to state obj
                                                                      // replace = simply add a new state or do we want to replace the current one with new one 
        void PopCurrent(); //to remove the current state from state Stack

        //// Actual modification of state stack
        void ProcessStateChange(); // to modify state stack

        std::unique_ptr<State>& GetCurrent(); // return an unique_ptr to current state object by reference
    
    };

} // namespace Engine