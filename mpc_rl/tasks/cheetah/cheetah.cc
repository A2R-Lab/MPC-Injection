// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/cheetah/cheetah.h"

#include <string>

#include <mujoco/mujoco.h>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string Cheetah::XmlPath() const { return GetModelPath("cheetah/task.xml"); }

std::string Cheetah::Name() const { return "Three-Legged Cheetah"; }

void Cheetah::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                   double* residual) const {
  int counter = 0;

  // ---------- Residual (0): control ----------
  mju_copy(residual + counter, data->ctrl, model->nu);
  counter += model->nu;

  // ---------- Residual (1): torso height ----------
  double height = SensorByName(model, data, "torso_position")[2];
  residual[counter++] = height - parameters_[0];

  // ---------- Residual (2): torso uprightness ----------
  double torso_up = SensorByName(model, data, "torso_zaxis")[2];
  residual[counter++] = torso_up - 1.0;

  // ---------- Residual (3): forward speed ----------
  double speed = SensorByName(model, data, "torso_subtreelinvel")[0];
  residual[counter++] = speed - parameters_[1];

  CheckSensorDim(model, counter);
}

}  // namespace mjpc
