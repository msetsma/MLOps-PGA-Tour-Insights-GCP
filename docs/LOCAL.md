# Command to run locally to set up my notebook

### Create a Virtual Environment
Run this command in your terminal (Command Prompt or PowerShell):

```bash
python -m venv .venv
```

### Activate the Virtual Environment
- **In PowerShell**:
   ```bash
   .\.venv\Scripts\Activate
   ```

- **In Command Prompt**:
   ```cmd
   .\.venv\Scripts\activate.bat
   ```

### Upgrade `pip` and Install Dependencies

```bash
pip install -r requirements.txt
```

### Register the Virtual Environment as a Jupyter Kernel

```bash
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

- `--name=venv`: Internal name for the kernel.
- `--display-name "Python (venv)"`: The name you'll see in Jupyter Notebook.


### Launch Jupyter Notebook:

```bash
jupyter notebook
```
- This opens a new browser window with the Jupyter Notebook interface.
- When creating a new notebook, you should see **"Python (venv)"** as an available kernel.
