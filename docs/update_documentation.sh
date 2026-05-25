conda init
conda activate nn-pytorch-torchdeq
make clean
sphinx-apidoc -f -H "Project content" -o ./source/ ../src
make html

