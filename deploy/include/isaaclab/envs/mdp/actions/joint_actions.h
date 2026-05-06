// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/manager/action_manager.h"

namespace isaaclab
{

class JointAction : public ActionTerm
{
public:
    JointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :ActionTerm(cfg, env)
    {
        if(cfg["joint_ids"].IsNull()) {
            _action_dim = env->robot->data.joint_ids_map.size();
        } else {
            _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
            _action_dim = _joint_ids.size();
        }
        _raw_actions.resize(_action_dim, 0.0f);
        _processed_actions.resize(_action_dim, 0.0f);
        _filtered_actions.resize(_action_dim, 0.0f);
        if(!cfg["scale"].IsNull()) {
            _scale = cfg["scale"].as<std::vector<float>>();
        }
        if(!cfg["offset"].IsNull()) {
            _offset = cfg["offset"].as<std::vector<float>>();
        }
        if(!cfg["clip"].IsNull()) {
            _clip = cfg["clip"].as<std::vector<std::vector<float> >>();
        }
        if(!cfg["low_pass_filter_cutoff_hz"].IsNull()) {
            const float cutoff_hz = cfg["low_pass_filter_cutoff_hz"].as<float>();
            if(cutoff_hz > 0.0f) {
                if(env->step_dt <= 0.0f) {
                    throw std::runtime_error("Action LPF requires positive env step_dt.");
                }
                constexpr float kPi = 3.14159265358979323846f;
                _lpf_alpha = 1.0f - std::exp(-2.0f * kPi * cutoff_hz * env->step_dt);
                _lpf_alpha = std::clamp(_lpf_alpha, 0.0f, 1.0f);
                _lpf_enabled = true;
            }
        }
    }

    virtual void process_actions(std::vector<float> actions)
    {
        // TODO: modify action by joint_ids
        _raw_actions = actions;
        for(int i(0); i<_action_dim; ++i)
        {
            if(!_scale.empty()) {
                _processed_actions[i] = _raw_actions[i] * _scale[i];
            } else {
                _processed_actions[i] = _raw_actions[i];
            }
            if(!_offset.empty()) {
                _processed_actions[i] += _offset[i];
            }
        }
        if(!_clip.empty())
        {
            for(int i(0); i<_action_dim; ++i) {
                _processed_actions[i] = std::clamp(_processed_actions[i], _clip[i][0], _clip[i][1]);
            }
        }

        if(_lpf_enabled)
        {
            if(!_lpf_initialized) {
                _filtered_actions = _processed_actions;
                _lpf_initialized = true;
            } else {
                for(int i(0); i<_action_dim; ++i) {
                    _filtered_actions[i] += _lpf_alpha * (_processed_actions[i] - _filtered_actions[i]);
                }
            }
            _processed_actions = _filtered_actions;
        }
    }


    int action_dim() 
    {
        return _action_dim;
    }

    std::vector<float> raw_actions() 
    {
        return _raw_actions;
    }
    
    std::vector<float> processed_actions() 
    {
        return _processed_actions;
    }

    void reset()
    {
        _lpf_initialized = false;
        process_actions(std::vector<float>(_action_dim, 0.0f));
    }

protected:
    int _action_dim;
    std::vector<int> _joint_ids;

    std::vector<float> _raw_actions;
    std::vector<float> _processed_actions;
    std::vector<float> _filtered_actions;

    std::vector<float> _scale;
    std::vector<float> _offset;
    std::vector<std::vector<float> > _clip;

    bool _lpf_enabled = false;
    bool _lpf_initialized = false;
    float _lpf_alpha = 1.0f;
};


class JointPositionAction : public JointAction
{
public:
    JointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

class JointVelocityAction : public JointAction
{
public:
    JointVelocityAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

REGISTER_ACTION(JointPositionAction);
REGISTER_ACTION(JointVelocityAction);

};
