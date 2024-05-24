<html>
<body>
<h2>Empirical Analysis for Unsupervised Universal Dependency Parse Tree Aggregation</h2>


<h3>Base Dependency Parsers Download Link</h3>

Please download the outputs of the participating dependency parsers of the CoNLL 2018 shared task from
https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2885.

<h3>Sequence of Execution</h3>
<ul>
<li>process_base_parsers.py: Python file to process official submissions of participating teams of CoNLL 2018 Shared Task</li>
<li>process_ground_truth.py: Python file to process ground truth</li>
<li>format_base_parser_outputs.py: Python file to format the output of base parsers</li>
<li>sentences_with_same_token_segmentation.py: This python file processes the output of the base parsers to find sentences with same token segmentation</li>
<li>create_binary_labeled_dataset.py: This python file creates binary labeled dataset for Ising model using the sentences with same token segmentation</li>
</ul>

<h3>Citation</h3>
Kindly cite our paper

```
@misc{kulkarni2024empirical,
      title={Empirical Analysis for Unsupervised Universal Dependency Parse Tree Aggregation}, 
      author={Adithya Kulkarni and Oliver Eulenstein and Qi Li},
      year={2024},
      eprint={2403.19183},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
</body>
</html>