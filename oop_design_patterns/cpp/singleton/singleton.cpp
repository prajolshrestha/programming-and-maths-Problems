#include "singleton.hpp"
// Initialize static class member to be a nullptr
Singleton* Singleton::instance = nullptr;

// Return single singleton instance. (Create only if instance doesnot exist)
Singleton *Singleton::getInstance(){
    // If instance == nullptr, create singleton object (ie, don't create until needed)
    if(!instance){
        instance = new Singleton();
    }
    return instance;
}