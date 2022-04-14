#ifndef KERAS2CPP_DELEGATES_HPP
#define KERAS2CPP_DELEGATES_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

namespace keras
{
  class Delegate;
  //class DelegateConv2D;
  //class DelegateFullyConnected;
  class DelegateSoftmax;
}

class keras::Delegate {
public:
    virtual ~Delegate() {}
    virtual std::vector<float> eval(std::vector<float> input, void *params) const {};
};

class keras::DelegateSoftmax : public Delegate {
public:
    DelegateSoftmax() : Delegate() {}
    ~DelegateSoftmax() {}
    std::vector<float> eval(std::vector<float> input, void *params) const {
        std::vector<float> output(input.size());

        return output;
    }
};

#endif /* KERAS2CPP_DELEGATES_HPP */
