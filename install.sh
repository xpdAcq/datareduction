echo "Do you want me to create a new empty environment for the installation? ([y]/n)"
read answer
if [ $answer != n ]
then
  echo "I will create a conda environment. What name would like to give it?"
  read env_name
  conda create -n $env_name --yes
else
  echo "What is the name of the existing environment where you would like to install this software?"
  read env_name
fi
echo "I will start installing the software in $env_name."
conda install -n $env_name --file requirements-conda.txt -c conda-forge --yes
conda run -n $env_name python -m pip install .
echo "I have finished the installtion. Please run the following conda to activate the environment."
echo ""
echo "    conda activate $env_name"
