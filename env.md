https://conda.io/docs/_downloads/conda-cheatsheet.pdf

### conda, jupyter env
`export PATH="/Users/yangshao/anaconda3/bin:$PATH"`

> no kernel exist for certain conda env:
```
conda install jupyter notebook
conda install nb_conda
conda install ipykernel
source activate myenv
python -m ipykernel install --user --name mykernel

```


```
conda info --envs
source activate py36(env)

conda list

conda install scikit-learn
conda install ipykernel
conda install jupyter
conda install tensorflow
conda install keras
conda install nb_conda
```

> see the python execution environment:

```
import sys
sys.executable
```

-----

> CondaError: Downloaded bytes did not match Content-Length:

```
conda config --show remote_read_timeout_secs
conda config --set remote_read_timeout_secs 300.0
```

-----

### tensorflow error summary

> Using TensorFlow 1.0.0:

`tf.python.control_flow_ops = tf`

> use tf.python_io in later versions:

`tf.python_io.control_flow_ops = tf`
