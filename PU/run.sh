# bin/bash

set -e 

# Train baseline on Wikipedia dataset
python train_wikipedia.py \
    --pos_file "Data/en_wiki_subset_statements_all_citations_sample.txt" \
    --neg_file "Data/en_wiki_subset_statements_no_citations_sample.txt" \
    --train_pct 0.8 \
    --n_gpu 1 \
    --log_interval 1 \
    --test_files "Data/media_en_statements_all_citations_sample.txt" "Data/media_en_statements_no_citations_sample.txt" \
    --test_files "Data/statements_cn_citations_sample.txt" "Data/statements_no_citations_sample.txt" \
    --seed 1000 \
    --model_dir models/wikipedia

# PU Learning on Wikipedia dataset 
python train_wikipedia_pu_learning.py \
  --pos_file "Data/en_wiki_subset_statements_all_citations_sample.txt" \
  --neg_file "Data/en_wiki_subset_statements_no_citations_sample.txt" \
  --train_pct 0.8 \
  --n_gpu 1 \
  --log_interval 1 \
  --test_files "Data/all_citations_sample.txt" "Data/no_citations_sample.txt" \
  --test_files "Data/statements_cn_citations_sample.txt" "Data/statements_no_citations_sample.txt" \
  --pretrained_model "models/wikipedia/model.pth" \
  --seed 1000 \
  --model_dir models/pu-wikipedia
  --indices_dir models/wikipedia

# PU Conversion on Wikipedia
python train_wikipedia_puc.py \
  --pos_file "data/wikipedia/english_citation_data/fa - featured articles/en_wiki_subset_statements_all_citations_sample.txt" \
  --neg_file "data/wikipedia/english_citation_data/fa - featured articles/en_wiki_subset_statements_no_citations_sample.txt" \
  --train_pct 0.8 \
  --n_gpu 1 \
  --log_interval 1 \
  --test_files "data/wikipedia/english_citation_data/rnd - random articles/all_citations_sample.txt" "data/wikipedia/english_citation_data/rnd - random articles/no_citations_sample.txt" \
  --test_files "data/wikipedia/english_citation_data/lqn - citation needed articles/statements_cn_citations_sample.txt" "data/wikipedia/english_citation_data/lqn - citation needed articles/statements_no_citations_sample.txt" \
  --pretrained_model "models/wikipedia/model.pth" \
  --seed 1000 \
  --model_dir models/puc-wikipedia
  --indices_dir models/wikipedia

# Baseline Pheme
python train_pheme.py \
  --pheme_dir "data/pheme/" \
  --train_pct 0.8 \
  --n_gpu 1 \
  --log_interval 1 \
  --exclude_splits ebola-essien gurlitt prince-toronto putinmissing \
  --seed 1000 \
  --model_dir models/pheme

# Pheme + PU
python {train_pheme_pu_learning.py|train_pheme_puc.py} \
  --pheme_dir "data/pheme/" \
  --train_pct 0.8 \
  --n_gpu 1 \
  --log_interval 1 \
  --exclude_splits ebola-essien gurlitt prince-toronto putinmissing \
  --seed 1000 \
  --model_dir models/{pheme-pu-solo|pheme-puc-solo} \
  --pretrained_pheme_model models/pheme \
  --indices_dir models/pheme

# Pheme + Wiki
python train_pheme.py \
  --pheme_dir "data/pheme/" \
  --train_pct 0.8 \
  --n_gpu 1 \
  --log_interval 1 \
  --exclude_splits ebola-essien gurlitt prince-toronto putinmissing \
  --seed 1000 \
  --model_dir models/{pheme-wiki|pheme-pu|pheme-puc} \
  --pretrained_model "models/{wikipedia|pu-wikipedia|puc-wikipedia}/model.pth"
  --indices_dir models/pheme
