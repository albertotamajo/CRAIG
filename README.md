# CRAIG
This repository implements the CRAIG coreset selection procedure described in <a href="https://arxiv.org/abs/1906.01827" target="_blank">Coresets for Data-efficient Training of Machine Learning Models</a> and  <a href="https://arxiv.org/abs/2306.01244" target="https://arxiv.org/abs/2306.01244">Towards Sustainable Learning: Coresets for Data-efficient Deep Learning</a>.

This implementation improves upon the official implementation in https://github.com/baharanm/craig and https://github.com/BigML-CS-UCLA/CREST:
- it uses an object-oriented approach for the implementation
- it provides a better documentation
- it guarantees that a coreset of a specific size is always output while in the original implementations there are cases where bigger coresets are extracted
- the indices of the samples selected for the coreset are sorted by their marginal gain in FL objective (largest gain first) for each class
- it handles errors better
