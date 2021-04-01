/*-------------------------------------------------------------------------------
 This file is part of diversityForest.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of divfor is distributed under MIT license and the
 R package "diversityForest" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#ifndef TREE_H_
#define TREE_H_

#include <vector>
#include <random>
#include <iostream>
#include <stdexcept>

#include "globals.h"
#include "Data.h"

namespace diversityForest {

class Tree {
public:
  Tree();

  // Create from loaded forest
  Tree(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
      std::vector<double>& split_values, std::vector<size_t>& split_types, std::vector<std::vector<size_t>>& split_multvarIDs, 
	  std::vector<std::vector<std::vector<bool>>>& split_directs, std::vector<std::vector<std::vector<double>>>& split_multvalues);

  virtual ~Tree() = default;

  Tree(const Tree&) = delete;
  Tree& operator=(const Tree&) = delete;

  void init(const Data* data, uint mtry, uint nsplits, uint npairs, double proptry, size_t dependent_varID, size_t num_samples, uint seed,
      std::vector<size_t>* deterministic_varIDs, std::vector<size_t>* split_select_varIDs,
      std::vector<double>* split_select_weights, ImportanceMode importance_mode, uint min_node_size,
      bool sample_with_replacement, bool memory_saving_splitting, SplitRule splitrule,
      std::vector<double>* case_weights, std::vector<size_t>* manual_inbag, bool keep_inbag,
      std::vector<double>* sample_fraction, double alpha, double minprop, bool holdout, uint num_random_splits,
      uint max_depth, std::vector<std::vector<size_t>>* promispairs, uint eim_mode, uint divfortype); // asdf

  virtual void allocateMemory() = 0;

  void grow(std::vector<double>* variable_importance);

  void predict(const Data* prediction_data, bool oob_prediction);
  void predictMultivariate(const Data *prediction_data, bool oob_prediction);

  void computePermutationImportance(std::vector<double>& forest_importance, std::vector<double>& forest_variance);
  void computePermutationImportanceMultivariate(std::vector<double> &forest_univ, std::vector<double> &forest_bivpooled, 
    std::vector<double> &forest_bivqual, std::vector<double> &forest_bivquant_ll,
	std::vector<double> &forest_bivquant_lh, std::vector<double> &forest_bivquant_hl, std::vector<double> &forest_bivquant_hh);

  void appendToFile(std::ofstream& file);
  virtual void appendToFileInternal(std::ofstream& file) = 0;

  const std::vector<std::vector<size_t>>& getChildNodeIDs() const {
    return child_nodeIDs;
  }
  const std::vector<double>& getSplitValues() const {
    return split_values;
  }
  
  const std::vector<size_t>& getSplitVarIDs() const {
    return split_varIDs;
  }

  const std::vector<size_t>& getSplitTypes() const {
    return split_types;
  }

  const std::vector<std::vector<size_t>>& getSplitMultVarIDs() const {
    return split_multvarIDs;
  }
  
  const std::vector<std::vector<std::vector<bool>>>& getSplitDirects() const {
    return split_directs;
  }

  const std::vector<std::vector<std::vector<double>>>& getSplitMultValues() const {
    return split_multvalues;
  }

  const std::vector<size_t>& getOobSampleIDs() const {
    return oob_sampleIDs;
  }
  
  size_t getNumSamplesOob() const {
    return num_samples_oob;
  }

  const std::vector<size_t>& getInbagCounts() const {
    return inbag_counts;
  }

protected:
  void createPossibleSplitVarSubset(std::vector<size_t>& result);
  void drawSplitsUnivariate(size_t nodeID, size_t n_triedsplits, std::vector<std::pair<size_t, double>>& sampled_varIDs_values); // asdf
  void drawSplitsMultivariate(size_t nodeID, size_t n_triedsplits, std::vector<size_t>& sampled_split_types, std::vector<std::vector<size_t>>& sampled_split_multvarIDs, std::vector<std::vector<std::vector<bool>>>& sampled_split_directs, std::vector<std::vector<std::vector<double>>>& sampled_split_multvalues); // asdf
  bool IsInRectangle(const Data* data, size_t sampleID, size_t split_type, std::vector<size_t>& split_multvarID, std::vector<std::vector<bool>>& split_direct, std::vector<std::vector<double>>& split_multvalue); // asdf	  
  
  bool splitNode(size_t nodeID);
  virtual bool splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) = 0;
  virtual bool splitNodeUnivariateInternal(size_t nodeID, std::vector<std::pair<size_t, double>> sampled_varIDs_values) = 0; // asdf
  virtual bool splitNodeMultivariateInternal(size_t nodeID, std::vector<size_t> sampled_split_types, std::vector<std::vector<size_t>> sampled_split_multvarIDs, std::vector<std::vector<std::vector<bool>>> sampled_split_directs, std::vector<std::vector<std::vector<double>>> sampled_split_multvalues) = 0; // asdf
   
  void createEmptyNode();
  virtual void createEmptyNodeInternal() = 0;
  void createEmptyNodeMultivariate();

  size_t dropDownSamplePermuted(size_t permuted_varID, size_t sampleID, size_t permuted_sampleID);
  
  size_t randomizedDropDownSample(std::vector<size_t> permuted_multvarID, size_t sampleID, size_t effect_type);
  bool randomAssignLeftChildNode(size_t nodeID);

  void permuteAndPredictOobSamples(size_t permuted_varID, std::vector<size_t>& permutations);

  void randomizedDropDownOobSamples(std::vector<size_t> permuted_multvarID, size_t effect_type);

  virtual double computePredictionAccuracyInternal() = 0;

  void bootstrap();
  void bootstrapWithoutReplacement();

  void bootstrapWeighted();
  void bootstrapWithoutReplacementWeighted();

  virtual void bootstrapClassWise();
  virtual void bootstrapWithoutReplacementClassWise();

  void setManualInbag();

  virtual void cleanUpInternal() = 0;

  size_t dependent_varID;
  uint mtry;
  uint nsplits; // asdf
  uint npairs;
  double proptry; // asdf
  
  // Number of samples (all samples, not only inbag for this tree)
  size_t num_samples;

  // Number of OOB samples
  size_t num_samples_oob;

  // Minimum node size to split, like in original RF nodes of smaller size can be produced
  uint min_node_size;

  // Weight vector for selecting possible split variables, one weight between 0 (never select) and 1 (always select) for each variable
  // Deterministic variables are always selected
  const std::vector<size_t>* deterministic_varIDs;
  const std::vector<size_t>* split_select_varIDs;
  const std::vector<double>* split_select_weights;

  // Bootstrap weights
  const std::vector<double>* case_weights;

  // Pre-selected bootstrap samples
  const std::vector<size_t>* manual_inbag;

  // Splitting variable for each node
  std::vector<size_t> split_varIDs;

  // Value to split at for each node, for now only binary split
  // For terminal nodes the prediction value is saved here
  std::vector<double> split_values;

  std::vector<size_t> split_types;
  std::vector<std::vector<size_t>> split_multvarIDs;
  std::vector<std::vector<std::vector<bool>>> split_directs;
  std::vector<std::vector<std::vector<double>>> split_multvalues;

  // Vector of left and right child node IDs, 0 for no child
  std::vector<std::vector<size_t>> child_nodeIDs;

  // All sampleIDs in the tree, will be re-ordered while splitting
  std::vector<size_t> sampleIDs;

  // For each node a vector with start and end positions
  std::vector<size_t> start_pos;
  std::vector<size_t> end_pos;

  // IDs of OOB individuals, sorted
  std::vector<size_t> oob_sampleIDs;

  std::vector<std::vector<size_t>>* promispairs;
  uint eim_mode;
  uint divfortype;
  
  // Holdout mode
  bool holdout;

  // Inbag counts
  bool keep_inbag;
  std::vector<size_t> inbag_counts;

  // Random number generator
  std::mt19937_64 random_number_generator;

  // Pointer to original data
  const Data* data;

  // Variable importance for all variables
  std::vector<double>* variable_importance;
  ImportanceMode importance_mode;

  // Effect importance:
  std::vector<double>* eim_univ;
  std::vector<double>* eim_bivpooled;
  std::vector<double>* eim_bivqual;
  std::vector<double>* eim_bivquant_ll;
  std::vector<double>* eim_bivquant_lh;
  std::vector<double>* eim_bivquant_hl;
  std::vector<double>* eim_bivquant_hh;

  // When growing here the OOB set is used
  // Terminal nodeIDs for prediction samples
  std::vector<size_t> prediction_terminal_nodeIDs;

  bool sample_with_replacement;
  const std::vector<double>* sample_fraction;

  bool memory_saving_splitting;
  SplitRule splitrule;
  double alpha;
  double minprop;
  uint num_random_splits;
  uint max_depth;
  uint depth;
  size_t last_left_nodeID;
};

} // namespace diversityForest

#endif /* TREE_H_ */
