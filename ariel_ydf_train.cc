#include <iostream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

// YDF dataset
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"

// YDF model
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"

// YDF learner
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"

using namespace yggdrasil_decision_forests;

absl::Status TrainRandomForest(const std::string& csv_path,
                               const std::string& label_column_name,
                               const std::string& output_model_dir) {
  // 1) Create a data specification for the CSV dataset.
  dataset::proto::DataSpecification data_spec;
  {
    // Infers the dataspec from the CSV data.
    std::cout << "Inferring DataSpec from CSV: " << csv_path << std::endl;
    // "csv:" prefix means a CSV dataset, as recognized by YDF.
    absl::Status status = dataset::CreateDataSpec(
        "csv:" + csv_path, /*validate=*/false,
        dataset::proto::DataSpecificationGuide(),  // No special guide
        &data_spec);
    if (!status.ok()) {
      return absl::InternalError("Could not create data spec: " +
                                 std::string(status.message()));
    }

    // (Optional) Print a summary of the DataSpec to stdout.
    std::cout << dataset::PrintHumanReadable(data_spec) << std::endl;
  }

  // 2) Set up the RandomForest configuration.
  model::proto::TrainingConfig train_config;
  train_config.set_learner("RANDOM_FOREST");
  train_config.set_task(model::proto::Task::CLASSIFICATION);  // or REGRESSION
  train_config.set_label(label_column_name);

  // If you need to override the number of threads, do so in the DeploymentConfig
  model::proto::DeploymentConfig deployment_config;
  deployment_config.set_num_threads(1);  // For single-threaded example

  // Example: setting random forest hyperparameters
  auto& rf_config = *train_config.MutableExtension(
      model::random_forest::proto::random_forest_config);
  rf_config.set_num_trees(1000);
  // Use -1 to mean "no maximum depth"
  rf_config.set_maximum_depth(-1);
  rf_config.set_bootstrap_training_dataset(true);
  rf_config.set_bootstrap_size_ratio(1.0);
  // etc.

  // 3) Create the learner from the config.
  std::unique_ptr<model::AbstractLearner> learner;
  {
    absl::Status get_learner_status = model::GetLearner(train_config, &learner);
    if (!get_learner_status.ok()) {
      return absl::InternalError("Could not create RandomForest learner: " +
                                 std::string(get_learner_status.message()));
    }
    // Optionally set deployment (resources) configuration:
    learner->SetDeploymentConfig(deployment_config);
  }

  // 4) (Optional) Link the dataspec to the training config to catch potential
  // mismatch or to skip re-inference. Not mandatory, but can be helpful.
  {
    absl::Status link_status =
        learner->LinkTrainingConfig(train_config, data_spec);
    if (!link_status.ok()) {
      return link_status;
    }
  }

  // 5) Train the model from disk-based dataset.  The "csv:" prefix means a
  //    CSV dataset recognized by YDF. E.g. "csv:/path/to/train.csv"
  absl::StatusOr<std::unique_ptr<model::AbstractModel>> model_or =
      learner->TrainWithStatus("csv:" + csv_path);
  if (!model_or.ok()) {
    return absl::InternalError("Training failed: " +
                               std::string(model_or.status().message()));
  }
  std::unique_ptr<model::AbstractModel> model = std::move(model_or.value());

  // 6) (Optional) Show basic info about the trained model
  std::cout << "Model trained. Summary:\n";
  std::cout << model->ShortDescription() << std::endl;

  // 7) Save the model to disk
  {
    absl::Status save_status = model::SaveModel(output_model_dir, *model);
    if (!save_status.ok()) {
      return absl::InternalError("Could not save model: " +
                                 std::string(save_status.message()));
    }
    std::cout << "Model saved to: " << output_model_dir << std::endl;
  }

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " train_data.csv label_column output_model_dir\n"
              << "Example: " << argv[0]
              << " /tmp/train.csv my_label /tmp/my_rf_model\n";
    return 1;
  }

  const std::string train_csv = argv[1];
  const std::string label_col = argv[2];
  const std::string model_out_dir = argv[3];

  // Train
  absl::Status status = TrainRandomForest(train_csv, label_col, model_out_dir);
  if (!status.ok()) {
    std::cerr << "Training failed: " << status.message() << std::endl;
    return 1;
  }

  std::cout << "Training complete.\n";
  return 0;
}