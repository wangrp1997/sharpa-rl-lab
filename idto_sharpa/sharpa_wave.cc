#include "examples/example_base.h"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/tree/rigid_body.h>
#include <drake/multibody/tree/spatial_inertia.h>
#include <drake/multibody/tree/unit_inertia.h>

// IDTO Allegro example, with this hand and the scale-0.5 cylinder.
// Kurtz, Castro, Onol, Lin, arXiv:2309.01813. Solver and contact numbers are
// unchanged from examples/allegro_hand. The hand model is right_sharpa_wave.

DEFINE_bool(test, false,
            "whether this example is being run in test mode, where we solve a "
            "simpler problem");

namespace idto {
namespace examples {
namespace sharpa_wave {

using drake::geometry::AddCompliantHydroelasticProperties;
using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::Cylinder;
using drake::geometry::ProximityProperties;
using drake::geometry::Rgba;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::math::RotationMatrixd;
using drake::multibody::CoulombFriction;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::UnitInertia;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using utils::FindIdtoResource;

constexpr double kRadius = 0.02;
constexpr double kLength = 0.032;
constexpr double kMass = 0.05;

const std::array<const char*, 22> kJoints = {
    "right_thumb_CMC_FE",  "right_thumb_CMC_AA", "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",  "right_thumb_IP",     "right_index_MCP_FE",
    "right_index_MCP_AA",  "right_index_PIP",    "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA","right_middle_PIP",
    "right_middle_DIP",    "right_ring_MCP_FE",  "right_ring_MCP_AA",
    "right_ring_PIP",      "right_ring_DIP",     "right_pinky_CMC",
    "right_pinky_MCP_FE",  "right_pinky_MCP_AA", "right_pinky_PIP",
    "right_pinky_DIP",
};

void CheckPositionOrder(const MultibodyPlant<double>& plant) {
  const std::vector<std::string> names = plant.GetPositionNames();
  const std::array<const char*, 7> tail = {"qw", "qx", "qy", "qz", "x", "y", "z"};
  std::string listing;
  for (const std::string& name : names) {
    listing += name + "\n";
  }
  if (names.size() != kJoints.size() + tail.size()) {
    throw std::runtime_error("Sharpa position count " + std::to_string(names.size()) +
                             "\n" + listing);
  }
  for (size_t i = 0; i < kJoints.size(); ++i) {
    if (names[i].find(kJoints[i]) == std::string::npos) {
      throw std::runtime_error(std::string("joint order mismatch at ") + kJoints[i] +
                               "\n" + listing);
    }
  }
  for (size_t i = 0; i < tail.size(); ++i) {
    const std::string& name = names[kJoints.size() + i];
    const std::string suffix = tail[i];
    if (name.size() < suffix.size() ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
      throw std::runtime_error("cylinder coordinate mismatch\n" + listing);
    }
  }
}

void AddSharpaScene(MultibodyPlant<double>* plant, bool for_simulation) {
  const drake::Vector4<double> blue(0.2, 0.3, 0.6, 1.0);
  const drake::Vector4<double> black(0.0, 0.0, 0.0, 1.0);

  const std::string urdf =
      FindIdtoResource("idto/models/right_sharpa_wave/right_sharpa_wave.urdf");
  Parser(plant).AddModels(urdf);
  const Quaterniond q_hand(0.819152, 0.0, -0.5735764, 0.0);
  const RigidTransformd X_hand(RotationMatrixd(q_hand), Vector3d(0.0, 0.0, 0.5));
  plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("right_hand_C_MC"),
                    X_hand);

  ModelInstanceIndex cylinder_idx = plant->AddModelInstance("cylinder");
  const SpatialInertia<double> inertia(
      kMass, Vector3d::Zero(),
      UnitInertia<double>::SolidCylinder(kRadius, kLength, Vector3d::UnitZ()));
  const RigidBody<double>& cylinder =
      plant->AddRigidBody("cylinder", cylinder_idx, inertia);
  plant->RegisterVisualGeometry(cylinder, RigidTransformd::Identity(),
                                Cylinder(kRadius, kLength), "cylinder_visual", blue);

  if (for_simulation) {
    ProximityProperties proximity;
    AddContactMaterial(3.0, {}, CoulombFriction<double>(1.0, 1.0), &proximity);
    AddCompliantHydroelasticProperties(0.01, 5e5, &proximity);
    plant->RegisterCollisionGeometry(cylinder, RigidTransformd::Identity(),
                                     Cylinder(kRadius, kLength), "cylinder_collision",
                                     proximity);
    const RigidTransformd X_ground(Vector3d(0.0, 0.0, -5.0));
    plant->RegisterCollisionGeometry(plant->world_body(), X_ground, Box(25, 25, 10),
                                     "ground", CoulombFriction<double>(1.0, 1.0));
  } else {
    plant->RegisterCollisionGeometry(cylinder, RigidTransformd::Identity(),
                                     Cylinder(kRadius, kLength), "cylinder_collision",
                                     CoulombFriction<double>(1.0, 1.0));
  }

  const RigidTransformd Xx(RollPitchYawd(0, M_PI_2, 0), Vector3d(kRadius / 2, 0, 0));
  const RigidTransformd Xy(RollPitchYawd(M_PI_2, 0, 0), Vector3d(0, kRadius / 2, 0));
  const RigidTransformd Xz(Vector3d(0, 0, kLength / 2));
  plant->RegisterVisualGeometry(cylinder, Xx, Cylinder(0.001, kRadius), "cylinder_axis_x",
                                drake::Vector4<double>(1.0, 0.0, 0.0, 1.0));
  plant->RegisterVisualGeometry(cylinder, Xy, Cylinder(0.001, kRadius), "cylinder_axis_y",
                                drake::Vector4<double>(0.0, 1.0, 0.0, 1.0));
  plant->RegisterVisualGeometry(cylinder, Xz, Cylinder(0.001, kLength), "cylinder_axis_z",
                                black);
}

class SharpaWaveExample : public TrajOptExample {
 public:
  SharpaWaveExample() {
    const Vector3d camera_pose(0.3, 0.0, 0.9);
    const Vector3d target_pose(-0.09, -0.01, 0.62);
    meshcat_->SetCameraPose(camera_pose, target_pose);
  }

 private:
  void UpdateCustomMeshcatElements(const TrajOptExampleParams& options) const final {
    const Vector3d target_position = options.q_nom_end.tail(3);
    const RotationMatrixd target_orientation(Quaterniond(
        options.q_nom_end[22], options.q_nom_end[23], options.q_nom_end[24],
        options.q_nom_end[25]));
    meshcat_->SetTransform("/desired_pose",
                           RigidTransformd(target_orientation, target_position));
  }

  void CreatePlantModel(MultibodyPlant<double>* plant) const final {
    AddSharpaScene(plant, false);
  }

  void CreatePlantModelForSimulation(MultibodyPlant<double>* plant) const final {
    AddSharpaScene(plant, true);
  }
};

}  // namespace sharpa_wave
}  // namespace examples
}  // namespace idto

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  {
    drake::multibody::MultibodyPlant<double> plant(0.0);
    drake::geometry::SceneGraph<double> scene_graph;
    plant.RegisterAsSourceForSceneGraph(&scene_graph);
    idto::examples::sharpa_wave::AddSharpaScene(&plant, false);
    plant.Finalize();
    idto::examples::sharpa_wave::CheckPositionOrder(plant);
  }

  idto::examples::sharpa_wave::SharpaWaveExample example;
  example.RunExample("idto/examples/sharpa_wave/sharpa_wave.yaml", FLAGS_test);
  return 0;
}
