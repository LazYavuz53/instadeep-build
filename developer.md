# Getting started

1. **Create a new environment**
   ```powershell
   python -m venv myenv
   ```

3. **Activate the environment**
   ```powershell
   .\myenv\Scripts\Activate
   ```
4. Pre-commit Setup
We use pre-commit hooks to check our code before each commit using the hooks defined in `.pre-commit-config.yaml`. First, please install the `pre-commit` package:

`pip install pre-commit`

After the install pre-commit should be executable from the terminal. If this is not the case, please adjust your PATH as follows:

* Adjust your bashrc to automatically update your path: `echo "export PATH=\$PATH:~/.local/bin/" >> ~/.bashrc`
* Update path of current terminal: source ~/.bashrc
* Run `which pre-commit` to verify that the tool is available

To setup the hooks, run `pre-commit install` the pre-commit hooks. The hooks will now be automatically run on every commit.
To test the hooks, run `pre-commit run --all-files`.


4. **Deactivate when finished**
   ```powershell
   deactivate
   ```
