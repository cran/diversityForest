/*-------------------------------------------------------------------------------
 This file is part of diversityForest.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of divfor is distributed under MIT license and the
 R package "diversityForest" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#include <iterator>

#include <Rcpp.h>

#include "Tree.h"
#include "utility.h"

namespace diversityForest {

Tree::Tree() :
    dependent_varID(0), mtry(0), max_triedsplits(0), proptry(0.0), num_samples(0), num_samples_oob(0), min_node_size(0), deterministic_varIDs(0), split_select_varIDs(
        0), split_select_weights(0), case_weights(0), manual_inbag(0), oob_sampleIDs(0), holdout(false), keep_inbag(
        false), data(0), variable_importance(0), importance_mode(DEFAULT_IMPORTANCE_MODE), sample_with_replacement(
        true), sample_fraction(0), memory_saving_splitting(false), splitrule(DEFAULT_SPLITRULE), alpha(DEFAULT_ALPHA), minprop(
        DEFAULT_MINPROP), num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(DEFAULT_MAXDEPTH), depth(0), last_left_nodeID(0) { // asdf
}

Tree::Tree(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
    std::vector<double>& split_values) :
    dependent_varID(0), mtry(0), max_triedsplits(0), proptry(0.0), num_samples(0), num_samples_oob(0), min_node_size(0), deterministic_varIDs(0), split_select_varIDs(
        0), split_select_weights(0), case_weights(0), manual_inbag(0), split_varIDs(split_varIDs), split_values(
        split_values), child_nodeIDs(child_nodeIDs), oob_sampleIDs(0), holdout(false), keep_inbag(false), data(0), variable_importance(
        0), importance_mode(DEFAULT_IMPORTANCE_MODE), sample_with_replacement(true), sample_fraction(0), memory_saving_splitting(
        false), splitrule(DEFAULT_SPLITRULE), alpha(DEFAULT_ALPHA), minprop(DEFAULT_MINPROP), num_random_splits(
        DEFAULT_NUM_RANDOM_SPLITS), max_depth(DEFAULT_MAXDEPTH), depth(0), last_left_nodeID(0) { // asdf
}

void Tree::init(const Data* data, uint mtry, uint max_triedsplits, double proptry, size_t dependent_varID, size_t num_samples, uint seed,
    std::vector<size_t>* deterministic_varIDs, std::vector<size_t>* split_select_varIDs,
    std::vector<double>* split_select_weights, ImportanceMode importance_mode, uint min_node_size,
    bool sample_with_replacement, bool memory_saving_splitting, SplitRule splitrule, std::vector<double>* case_weights,
    std::vector<size_t>* manual_inbag, bool keep_inbag, std::vector<double>* sample_fraction, double alpha,
    double minprop, bool holdout, uint num_random_splits, uint max_depth) { // asdf

  this->data = data;
  this->mtry = mtry;
  this->dependent_varID = dependent_varID;
  this->num_samples = num_samples;
  this->memory_saving_splitting = memory_saving_splitting;
  this->max_triedsplits = max_triedsplits; // asdf
  this->proptry = proptry; // asdf
  
  // Create root node, assign bootstrap sample and oob samples
  child_nodeIDs.push_back(std::vector<size_t>());
  child_nodeIDs.push_back(std::vector<size_t>());
  createEmptyNode();

  // Initialize random number generator and set seed
  random_number_generator.seed(seed);

  this->deterministic_varIDs = deterministic_varIDs;
  this->split_select_varIDs = split_select_varIDs;
  this->split_select_weights = split_select_weights;
  this->importance_mode = importance_mode;
  this->min_node_size = min_node_size;
  this->sample_with_replacement = sample_with_replacement;
  this->splitrule = splitrule;
  this->case_weights = case_weights;
  this->manual_inbag = manual_inbag;
  this->keep_inbag = keep_inbag;
  this->sample_fraction = sample_fraction;
  this->holdout = holdout;
  this->alpha = alpha;
  this->minprop = minprop;
  this->num_random_splits = num_random_splits;
  this->max_depth = max_depth;
}

void Tree::grow(std::vector<double>* variable_importance) {
  // Allocate memory for tree growing
  allocateMemory();

  this->variable_importance = variable_importance;

// Bootstrap, dependent if weighted or not and with or without replacement
  if (!case_weights->empty()) {
    if (sample_with_replacement) {
      bootstrapWeighted();
    } else {
      bootstrapWithoutReplacementWeighted();
    }
  } else if (sample_fraction->size() > 1) {
    if (sample_with_replacement) {
      bootstrapClassWise();
    } else {
      bootstrapWithoutReplacementClassWise();
    }
  } else if (!manual_inbag->empty()) {
    setManualInbag();
  } else {
    if (sample_with_replacement) {
      bootstrap();
    } else {
      bootstrapWithoutReplacement();
    }
  }

  // Init start and end positions
  start_pos[0] = 0;
  end_pos[0] = sampleIDs.size();

  // While not all nodes terminal, split next node
  size_t num_open_nodes = 1;
  size_t i = 0;
  depth = 0;
  while (num_open_nodes > 0) {
    // Split node
    bool is_terminal_node = splitNode(i);
    if (is_terminal_node) {
      --num_open_nodes;
    } else {
      ++num_open_nodes;
      if (i >= last_left_nodeID) {
        // If new level, increase depth
        // (left_node saves left-most node in current level, new level reached if that node is splitted)
        last_left_nodeID = split_varIDs.size() - 2;
        ++depth;
      }
    }
    ++i;
  }

  // Delete sampleID vector to save memory
  sampleIDs.clear();
  sampleIDs.shrink_to_fit();
  cleanUpInternal();
}

void Tree::predict(const Data* prediction_data, bool oob_prediction) {

  size_t num_samples_predict;
  if (oob_prediction) {
    num_samples_predict = num_samples_oob;
  } else {
    num_samples_predict = prediction_data->getNumRows();
  }

  prediction_terminal_nodeIDs.resize(num_samples_predict, 0);

// For each sample start in root, drop down the tree and return final value
  for (size_t i = 0; i < num_samples_predict; ++i) {
    size_t sample_idx;
    if (oob_prediction) {
      sample_idx = oob_sampleIDs[i];
    } else {
      sample_idx = i;
    }
    size_t nodeID = 0;
    while (1) {

      // Break if terminal node
      if (child_nodeIDs[0][nodeID] == 0 && child_nodeIDs[1][nodeID] == 0) {
        break;
      }

      // Move to child
      size_t split_varID = split_varIDs[nodeID];

      double value = prediction_data->get(sample_idx, split_varID);
      if (prediction_data->isOrderedVariable(split_varID)) {
        if (value <= split_values[nodeID]) {
          // Move to left child
          nodeID = child_nodeIDs[0][nodeID];
        } else {
          // Move to right child
          nodeID = child_nodeIDs[1][nodeID];
        }
      } else {
        size_t factorID = floor(value) - 1;
        size_t splitID = floor(split_values[nodeID]);

        // Left if 0 found at position factorID
        if (!(splitID & (1 << factorID))) {
          // Move to left child
          nodeID = child_nodeIDs[0][nodeID];
        } else {
          // Move to right child
          nodeID = child_nodeIDs[1][nodeID];
        }
      }
    }

    prediction_terminal_nodeIDs[i] = nodeID;
  }
}

void Tree::computePermutationImportance(std::vector<double>& forest_importance, std::vector<double>& forest_variance) {

  size_t num_independent_variables = data->getNumCols() - data->getNoSplitVariables().size();

// Compute normal prediction accuracy for each tree. Predictions already computed..
  double accuracy_normal = computePredictionAccuracyInternal();

  prediction_terminal_nodeIDs.clear();
  prediction_terminal_nodeIDs.resize(num_samples_oob, 0);

// Reserve space for permutations, initialize with oob_sampleIDs
  std::vector<size_t> permutations(oob_sampleIDs);

// Randomly permute for all independent variables
  for (size_t i = 0; i < num_independent_variables; ++i) {

    // Skip no split variables
    size_t varID = i;
    for (auto& skip : data->getNoSplitVariables()) {
      if (varID >= skip) {
        ++varID;
      }
    }

    // Permute and compute prediction accuracy again for this permutation and save difference
    permuteAndPredictOobSamples(varID, permutations);
    double accuracy_permuted = computePredictionAccuracyInternal();
    double accuracy_difference = accuracy_normal - accuracy_permuted;
    forest_importance[i] += accuracy_difference;

    // Compute variance
    if (importance_mode == IMP_PERM_BREIMAN) {
      forest_variance[i] += accuracy_difference * accuracy_difference;
    } else if (importance_mode == IMP_PERM_LIAW) {
      forest_variance[i] += accuracy_difference * accuracy_difference * num_samples_oob;
    }
  }
}

void Tree::appendToFile(std::ofstream& file) {

// Save general fields
  saveVector2D(child_nodeIDs, file);
  saveVector1D(split_varIDs, file);
  saveVector1D(split_values, file);

// Call special functions for subclasses to save special fields.
  appendToFileInternal(file);
}

void Tree::createPossibleSplitVarSubset(std::vector<size_t>& result) {

  size_t num_vars = data->getNumCols();

  // For corrected Gini importance add dummy variables
  if (importance_mode == IMP_GINI_CORRECTED) {
    num_vars += data->getNumCols() - data->getNoSplitVariables().size();
  }

  // Randomly add non-deterministic variables (according to weights if needed)
  if (split_select_weights->empty()) {
    if (deterministic_varIDs->empty()) {
      drawWithoutReplacementSkip(result, random_number_generator, num_vars, data->getNoSplitVariables(), mtry);
    } else {
      std::vector<size_t> skip;
      std::copy(data->getNoSplitVariables().begin(), data->getNoSplitVariables().end(),
          std::inserter(skip, skip.end()));
      std::copy(deterministic_varIDs->begin(), deterministic_varIDs->end(), std::inserter(skip, skip.end()));
      std::sort(skip.begin(), skip.end());
      drawWithoutReplacementSkip(result, random_number_generator, num_vars, skip, mtry);
    }
  } else {
    drawWithoutReplacementWeighted(result, random_number_generator, *split_select_varIDs, mtry, *split_select_weights);
  }

  // Always use deterministic variables
  std::copy(deterministic_varIDs->begin(), deterministic_varIDs->end(), std::inserter(result, result.end()));
}

// asdf: New function.
// This function determines the number of split points to sample,
// that is, the number of all possible split points times the value
// of 'proprtry', but at maximum the number given by 'max_triedsplits'.
void Tree::drawSplitsUnivariate(size_t nodeID, size_t n_triedsplits, std::vector<std::pair<size_t, double>>& sampled_varIDs_values) { // katze
  
  // Determine the indices of all variables
  size_t num_vars = data->getNumCols();
  
  // For corrected Gini importance add dummy variables
  if (importance_mode == IMP_GINI_CORRECTED) {
    num_vars += data->getNumCols() - data->getNoSplitVariables().size();
  }
  
  
  
  // std::vector<size_t> all_varIDs; // Auskommentiert, weil das nicht funktioniert hat.
  // all_varIDs.reserve(num_vars - data->getNoSplitVariables().size());
  std::vector<int> all_varIDs(num_vars - data->getNoSplitVariables().size());
  std::iota (all_varIDs.begin(), all_varIDs.end(), 1);
  
  // Cycle through all variables, count their numbers of split
  // points and add these up
  size_t n_splitstotal = 0;
  	  size_t n_triedsplitscandidate;
  for (auto& varID : all_varIDs) {
    
    // Create possible split values for variable 'varID'
    std::vector<double> possible_split_values;
    data->getAllValues(possible_split_values, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);
   
   // Rcpp::Rcout << " " << std::endl;
   	// Rcpp::Rcout << "Splitpunkte: " << std::endl;
   // std::copy(begin(possible_split_values), end(possible_split_values), std::ostream_iterator<double>(std::cout, " "));
   // Rcpp::Rcout << " " << std::endl;
   
    // Add the number of split values up to the total number of
    // split values:
    n_splitstotal += possible_split_values.size() - 1;
	
	n_triedsplitscandidate = (size_t)((double) n_splitstotal * proptry + 0.5);

	   if (n_triedsplitscandidate > n_triedsplits) {
            break;
        }

  } 
  
    // Die nachfolgende Option soll verwendet werden, zu verhindern, dass versucht wird, zu splitten,
  // obwohl gar keine Splitpunkte mehr da sind.
  // Evtl kann man das auch weglassen, je nachdem, ob das vielleicht
  // von dem danach folgenden Code auch schon gemacht wird.
  // if (n_splitstotal == 0) {
  //    return true;
  //}

  // If the calculated number of splits to sample
  // is larger than the maximum number of splits to sample 'max_triedsplits',
  // use 'max_triedsplits':
  n_triedsplits = std::min(n_triedsplits, n_triedsplitscandidate);
  
  // Rcpp::Rcout << "ntriedsplits: " << n_triedsplits << std::endl;
  
  // If the result of n_triedsplits was zero, it is set to one,
  // because at least one split point should be used.
  //if (n_triedsplits == 0) {
    //n_triedsplits = 1;
//	return true;
  //}
  
  // Rcpp::Rcout << "ntriedsplits: " << n_triedsplits << std::endl;
  // Rcpp::Rcout << " " << std::endl;
  // Rcpp::Rcout << " " << std::endl;
  // Rcpp::Rcout << " " << std::endl;

    if (n_triedsplits > 0) {
    
  sampled_varIDs_values.reserve(n_triedsplits);
  
  std::uniform_int_distribution<size_t> unif_distvarID(1, num_vars - data->getNoSplitVariables().size());
  
  size_t drawnvarID;
  double drawnvalue;
  
  for (size_t i = 0; i < n_triedsplits; ++i) { 
    
    std::pair <size_t, double> drawnpair;   
    bool pairnotfound = false;
	
    do {
     
	  // Rcpp::Rcout << "Schon drin: " << pairnotfound << std::endl;
	  	  
      drawnvarID = unif_distvarID(random_number_generator);
      
	  // Rcpp::Rcout << "gezogene Variable: " << drawnvarID << std::endl;
	  
      // Create possible split values for variable 'varID'
      std::vector<double> possible_split_values;
      data->getAllValues(possible_split_values, sampleIDs, drawnvarID, start_pos[nodeID], end_pos[nodeID]);
      
	  pairnotfound = possible_split_values.size() < 2;
	  
	  if(!pairnotfound) {
	   // Rcpp::Rcout << "Splitpunkte: " << std::endl;
       // std::copy(begin(possible_split_values), end(possible_split_values), std::ostream_iterator<double>(std::cout, " "));
	   // Rcpp::Rcout << " " << std::endl;
	 
	 	  std::vector<double> all_mid_points(possible_split_values.size()-1);
  // Compute decrease of impurity for each possible split
  for (size_t i = 0; i < possible_split_values.size()-1; ++i) {
      all_mid_points[i] = (possible_split_values[i] + possible_split_values[i + 1]) / 2;
    }
	 
      std::uniform_int_distribution<size_t> unif_distvalue(0, all_mid_points.size() - 1);
      
      drawnvalue = all_mid_points[unif_distvalue(random_number_generator)];
      
	  // Rcpp::Rcout << "gezogener Splitpunkt: " << drawnvalue << std::endl;
	  
      drawnpair = std::make_pair(drawnvarID, drawnvalue);
	  
	  pairnotfound = std::find(sampled_varIDs_values.begin(), sampled_varIDs_values.end(), drawnpair) != sampled_varIDs_values.end();
	  
	  }
      
    } while (pairnotfound);
    
    sampled_varIDs_values.push_back(drawnpair);
	
  }
  
 //   std::vector<size_t> gezogenevars;
 // std::vector<double> gezogenepunkte;
 //  for (size_t i = 0; i < sampled_varIDs_values.size(); ++i) {
//	gezogenevars.push_back(std::get<0>(sampled_varIDs_values[i]));
//	gezogenepunkte.push_back(std::get<1>(sampled_varIDs_values[i]));
//	}
	
	// Rcpp::Rcout << " " << std::endl;
// Rcpp::Rcout << " " << std::endl;
// Rcpp::Rcout << " " << std::endl;
// Rcpp::Rcout << "Gezogene Variablen:" << std::endl;
// std::copy(begin(gezogenevars), end(gezogenevars), std::ostream_iterator<size_t>(std::cout, " "));
// Rcpp::Rcout << " " << std::endl;
// Rcpp::Rcout << "Gezogene Punkte:" << std::endl;
// std::copy(begin(gezogenepunkte), end(gezogenepunkte), std::ostream_iterator<double>(std::cout, " "));
// Rcpp::Rcout << " " << std::endl;
// std::get<0>(sampled_varIDs_values[42]);
// std::get<1>(sampled_varIDs_values[42]);
    
  }

}

bool Tree::splitNode(size_t nodeID) {

  // Select random subset of variables to possibly split at  \\ Roman
  // std::vector<size_t> possible_split_varIDs;
  // createPossibleSplitVarSubset(possible_split_varIDs);

  // Draw the variables and the candidate splits - after performing this step,
  // sampled_varIDs_values will contain the variables and candidate splits:
  size_t n_triedsplits = (size_t) max_triedsplits; // asdf
  std::vector<std::pair<size_t, double>> sampled_varIDs_values;
  drawSplitsUnivariate(nodeID, n_triedsplits, sampled_varIDs_values); // asdf

  
  // Hier gehts weiter:
  
  // WICHTIG: Obige Funktion drawSplitsUnivariate passt noch nicht, weil
  // da noch nicht die eigentlichen Splitpunkte gesameplet werden, welche
  // zwischen den uniquen Werten liegen, sondern die uniquen Werte.
  // --> To Do: Ausprobieren, ob der Standardcode von ranger auch
  // genauso funktioniert, wenn man in den Funktionen findBestSplitValueSmallQ
  // und findBestSplitValueLargeQ nicht die possible_split_values
  // verewndet, sondern stattdessen gleich die Mittelpunkte zwischen
  // diesen possible_split_values. Wenn das so ist, kann ich in
  // drawSplitsUnivariate gleich die Mittelpunkte zwischen den 
  // possible_split_values sampeln und die dann in den entsprechend
  // abgeaenderten Funktionen findBestSplitValueSmallQ und
  // findBestSplitValueLargeQ verwenden.
  // Das kann ich herausfinden, in dem ich es einfach ausprobiere.
  
  // Oben wurden die Splitpunkte gesampelt. Wenn es entweder keinen einzigen
  // Splitpunkt in den Variablen gibt oder wenn round(proptry * 'Anzahl Splitpunkte') = 0
  // ist, gibt die obige Funktion 'drawSplitsUnivariate' true zurueck, in welchem
  // Fall die Berechnung mit folgendem Ausdruck abgebrochen wird
  // und wird demnach in einem Endknoten gelandet sind.
  // Die mit obiger Funktion gesampelten Splitpunkte sind zudem
  // alle tatsaechlich Splitpunkte, d.h. asdf
  
  
  // if(nosplitpointsleft) {
  //  return true;
  // }
  
  // Perform the splitting:
  bool stop = splitNodeUnivariateInternal(nodeID, sampled_varIDs_values); // asdf
    
  // Call subclass method, sets split_varIDs and split_values
  // bool stop = splitNodeInternal(nodeID, possible_split_varIDs);
  if (stop) {
    // Terminal node
    return true;
  }

  size_t split_varID = split_varIDs[nodeID];
  double split_value = split_values[nodeID];

    // Rcpp::Rcout << " " << std::endl;
  // Rcpp::Rcout << "split_varID: " << split_varID << std::endl;
  // Rcpp::Rcout << "split_value: " << split_value << std::endl;
  // Rcpp::Rcout << " " << std::endl;
  
  // Save non-permuted variable for prediction
  split_varIDs[nodeID] = data->getUnpermutedVarID(split_varID);

  // Create child nodes
  size_t left_child_nodeID = split_varIDs.size();
  child_nodeIDs[0][nodeID] = left_child_nodeID;
  createEmptyNode();
  start_pos[left_child_nodeID] = start_pos[nodeID];

  size_t right_child_nodeID = split_varIDs.size();
  child_nodeIDs[1][nodeID] = right_child_nodeID;
  createEmptyNode();
  start_pos[right_child_nodeID] = end_pos[nodeID];

  // For each sample in node, assign to left or right child
  if (data->isOrderedVariable(split_varID)) {
    // Ordered: left is <= splitval and right is > splitval
    size_t pos = start_pos[nodeID];
    while (pos < start_pos[right_child_nodeID]) {
      size_t sampleID = sampleIDs[pos];
      if (data->get(sampleID, split_varID) <= split_value) {
        // If going to left, do nothing
        ++pos;
      } else {
        // If going to right, move to right end
        --start_pos[right_child_nodeID];
        std::swap(sampleIDs[pos], sampleIDs[start_pos[right_child_nodeID]]);
      }
    }
  } else {
    // Unordered: If bit at position is 1 -> right, 0 -> left
    size_t pos = start_pos[nodeID];
    while (pos < start_pos[right_child_nodeID]) {
      size_t sampleID = sampleIDs[pos];
      double level = data->get(sampleID, split_varID);
      size_t factorID = floor(level) - 1;
      size_t splitID = floor(split_value);

      // Left if 0 found at position factorID
      if (!(splitID & (1 << factorID))) {
        // If going to left, do nothing
        ++pos;
      } else {
        // If going to right, move to right end
        --start_pos[right_child_nodeID];
        std::swap(sampleIDs[pos], sampleIDs[start_pos[right_child_nodeID]]);
      }
    }
  }

  // End position of left child is start position of right child
  end_pos[left_child_nodeID] = start_pos[right_child_nodeID];
  end_pos[right_child_nodeID] = end_pos[nodeID];
  
  // No terminal node
  return false;
}

void Tree::createEmptyNode() {
  split_varIDs.push_back(0);
  split_values.push_back(0);
  child_nodeIDs[0].push_back(0);
  child_nodeIDs[1].push_back(0);
  start_pos.push_back(0);
  end_pos.push_back(0);

  createEmptyNodeInternal();
}

size_t Tree::dropDownSamplePermuted(size_t permuted_varID, size_t sampleID, size_t permuted_sampleID) {

// Start in root and drop down
  size_t nodeID = 0;
  while (child_nodeIDs[0][nodeID] != 0 || child_nodeIDs[1][nodeID] != 0) {

    // Permute if variable is permutation variable
    size_t split_varID = split_varIDs[nodeID];
    size_t sampleID_final = sampleID;
    if (split_varID == permuted_varID) {
      sampleID_final = permuted_sampleID;
    }

    // Move to child
    double value = data->get(sampleID_final, split_varID);
    if (data->isOrderedVariable(split_varID)) {
      if (value <= split_values[nodeID]) {
        // Move to left child
        nodeID = child_nodeIDs[0][nodeID];
      } else {
        // Move to right child
        nodeID = child_nodeIDs[1][nodeID];
      }
    } else {
      size_t factorID = floor(value) - 1;
      size_t splitID = floor(split_values[nodeID]);

      // Left if 0 found at position factorID
      if (!(splitID & (1 << factorID))) {
        // Move to left child
        nodeID = child_nodeIDs[0][nodeID];
      } else {
        // Move to right child
        nodeID = child_nodeIDs[1][nodeID];
      }
    }

  }
  return nodeID;
}

void Tree::permuteAndPredictOobSamples(size_t permuted_varID, std::vector<size_t>& permutations) {

// Permute OOB sample
//std::vector<size_t> permutations(oob_sampleIDs);
  std::shuffle(permutations.begin(), permutations.end(), random_number_generator);

// For each sample, drop down the tree and add prediction
  for (size_t i = 0; i < num_samples_oob; ++i) {
    size_t nodeID = dropDownSamplePermuted(permuted_varID, oob_sampleIDs[i], permutations[i]);
    prediction_terminal_nodeIDs[i] = nodeID;
  }
}

void Tree::bootstrap() {

// Use fraction (default 63.21%) of the samples
  size_t num_samples_inbag = (size_t) num_samples * (*sample_fraction)[0];

// Reserve space, reserve a little more to be save)
  sampleIDs.reserve(num_samples_inbag);
  oob_sampleIDs.reserve(num_samples * (exp(-(*sample_fraction)[0]) + 0.1));

  std::uniform_int_distribution<size_t> unif_dist(0, num_samples - 1);

// Start with all samples OOB
  inbag_counts.resize(num_samples, 0);

// Draw num_samples samples with replacement (num_samples_inbag out of n) as inbag and mark as not OOB
  for (size_t s = 0; s < num_samples_inbag; ++s) {
    size_t draw = unif_dist(random_number_generator);
    sampleIDs.push_back(draw);
    ++inbag_counts[draw];
  }

// Save OOB samples
  for (size_t s = 0; s < inbag_counts.size(); ++s) {
    if (inbag_counts[s] == 0) {
      oob_sampleIDs.push_back(s);
    }
  }
  num_samples_oob = oob_sampleIDs.size();

  if (!keep_inbag) {
    inbag_counts.clear();
    inbag_counts.shrink_to_fit();
  }
}

void Tree::bootstrapWeighted() {

// Use fraction (default 63.21%) of the samples
  size_t num_samples_inbag = (size_t) num_samples * (*sample_fraction)[0];

// Reserve space, reserve a little more to be save)
  sampleIDs.reserve(num_samples_inbag);
  oob_sampleIDs.reserve(num_samples * (exp(-(*sample_fraction)[0]) + 0.1));

  std::discrete_distribution<> weighted_dist(case_weights->begin(), case_weights->end());

// Start with all samples OOB
  inbag_counts.resize(num_samples, 0);

// Draw num_samples samples with replacement (n out of n) as inbag and mark as not OOB
  for (size_t s = 0; s < num_samples_inbag; ++s) {
    size_t draw = weighted_dist(random_number_generator);
    sampleIDs.push_back(draw);
    ++inbag_counts[draw];
  }

  // Save OOB samples. In holdout mode these are the cases with 0 weight.
  if (holdout) {
    for (size_t s = 0; s < (*case_weights).size(); ++s) {
      if ((*case_weights)[s] == 0) {
        oob_sampleIDs.push_back(s);
      }
    }
  } else {
    for (size_t s = 0; s < inbag_counts.size(); ++s) {
      if (inbag_counts[s] == 0) {
        oob_sampleIDs.push_back(s);
      }
    }
  }
  num_samples_oob = oob_sampleIDs.size();

  if (!keep_inbag) {
    inbag_counts.clear();
    inbag_counts.shrink_to_fit();
  }
}

void Tree::bootstrapWithoutReplacement() {

// Use fraction (default 63.21%) of the samples
  size_t num_samples_inbag = (size_t) num_samples * (*sample_fraction)[0];
  shuffleAndSplit(sampleIDs, oob_sampleIDs, num_samples, num_samples_inbag, random_number_generator);
  num_samples_oob = oob_sampleIDs.size();

  if (keep_inbag) {
    // All observation are 0 or 1 times inbag
    inbag_counts.resize(num_samples, 1);
    for (size_t i = 0; i < oob_sampleIDs.size(); i++) {
      inbag_counts[oob_sampleIDs[i]] = 0;
    }
  }
}

void Tree::bootstrapWithoutReplacementWeighted() {

// Use fraction (default 63.21%) of the samples
  size_t num_samples_inbag = (size_t) num_samples * (*sample_fraction)[0];
  drawWithoutReplacementWeighted(sampleIDs, random_number_generator, num_samples - 1, num_samples_inbag, *case_weights);

// All observation are 0 or 1 times inbag
  inbag_counts.resize(num_samples, 0);
  for (auto& sampleID : sampleIDs) {
    inbag_counts[sampleID] = 1;
  }

// Save OOB samples. In holdout mode these are the cases with 0 weight.
  if (holdout) {
    for (size_t s = 0; s < (*case_weights).size(); ++s) {
      if ((*case_weights)[s] == 0) {
        oob_sampleIDs.push_back(s);
      }
    }
  } else {
    for (size_t s = 0; s < inbag_counts.size(); ++s) {
      if (inbag_counts[s] == 0) {
        oob_sampleIDs.push_back(s);
      }
    }
  }
  num_samples_oob = oob_sampleIDs.size();

  if (!keep_inbag) {
    inbag_counts.clear();
    inbag_counts.shrink_to_fit();
  }
}

void Tree::bootstrapClassWise() {
  // Empty on purpose (virtual function only implemented in classification and probability)
}

void Tree::bootstrapWithoutReplacementClassWise() {
  // Empty on purpose (virtual function only implemented in classification and probability)
}

void Tree::setManualInbag() {
  // Select observation as specified in manual_inbag vector
  sampleIDs.reserve(manual_inbag->size());
  inbag_counts.resize(num_samples, 0);
  for (size_t i = 0; i < manual_inbag->size(); ++i) {
    size_t inbag_count = (*manual_inbag)[i];
    if ((*manual_inbag)[i] > 0) {
      for (size_t j = 0; j < inbag_count; ++j) {
        sampleIDs.push_back(i);
      }
      inbag_counts[i] = inbag_count;
    } else {
      oob_sampleIDs.push_back(i);
    }
  }
  num_samples_oob = oob_sampleIDs.size();

  // Shuffle samples
  std::shuffle(sampleIDs.begin(), sampleIDs.end(), random_number_generator);

  if (!keep_inbag) {
    inbag_counts.clear();
    inbag_counts.shrink_to_fit();
  }
}

} // namespace diversityForest
