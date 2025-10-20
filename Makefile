PY := /usr/local/bin/python3.10
RUN := source .venv/bin/activate

.PHONY: venv install data train eval infer freeze clean

venv:
	$(PY) -m venv .venv

install: venv
	. .venv/bin/activate && pip install --upgrade pip setuptools wheel
	. .venv/bin/activate && pip install -r requirements.txt

data:
	bash scripts/download_data.sh

train:
	. .venv/bin/activate && python -m fruits360.train

eval:
	. .venv/bin/activate && python -m fruits360.eval

infer:
	. .venv/bin/activate && python -m fruits360.infer --image "$(IMG)"

freeze:
	. .venv/bin/activate && pip freeze | sed '/^pkg_resources==/d' > requirements.lock.txt
	@echo "Wrote requirements.lock.txt"

clean:
	rm -rf .venv artifacts

