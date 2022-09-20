python -m scripts.evilslime.train -a argfiles/evilslime/cartpole-v1.json -t base
python -m scripts.evilslime.evaluate -a argfiles/evilslime/cartpole-v1.json -t base
python -m scripts.evilslime.featureselect -a argfiles/evilslime/cartpole-v1.json -t base

python -m scripts.evilslime.train -a argfiles/evilslime/acrobot-v1.json -t base
python -m scripts.evilslime.evaluate -a argfiles/evilslime/acrobot-v1.json -t base
python -m scripts.evilslime.featureselect -a argfiles/evilslime/acrobot-v1.json -t base

python -m scripts.evilslime.train -a argfiles/evilslime/pendulum-v1.json -t base
python -m scripts.evilslime.evaluate -a argfiles/evilslime/pendulum-v1.json -t base
python -m scripts.evilslime.featureselect -a argfiles/evilslime/pendulum-v1.json -t base

python -m scripts.evilslime.train -a argfiles/evilslime/lunarlander-v2.json -t base
python -m scripts.evilslime.evaluate -a argfiles/evilslime/lunarlander-v2.json -t base
python -m scripts.evilslime.featureselect -a argfiles/evilslime/lunarlander-v2.json -t base

python -m scripts.evilslime.train -a argfiles/evilslime/bipedalwalker-v3.json -t base
python -m scripts.evilslime.evaluate -a argfiles/evilslime/bipedalwalker-v3.json -t base
python -m scripts.evilslime.featureselect -a argfiles/evilslime/bipedalwalker-v3.json -t base

python -m scripts.evilslime.train -a argfiles/evilslime/cartpole-v1.json -t l1
python -m scripts.evilslime.evaluate -a argfiles/evilslime/cartpole-v1.json -t l1

python -m scripts.evilslime.train -a argfiles/evilslime/acrobot-v1.json -t l1
python -m scripts.evilslime.evaluate -a argfiles/evilslime/acrobot-v1.json -t l1

python -m scripts.evilslime.train -a argfiles/evilslime/pendulum-v1.json -t l1
python -m scripts.evilslime.evaluate -a argfiles/evilslime/pendulum-v1.json -t l1

python -m scripts.evilslime.train -a argfiles/evilslime/lunarlander-v2.json -t l1
python -m scripts.evilslime.evaluate -a argfiles/evilslime/lunarlander-v2.json -t l1

python -m scripts.evilslime.train -a argfiles/evilslime/bipedalwalker-v3.json -t l1
python -m scripts.evilslime.evaluate -a argfiles/evilslime/bipedalwalker-v3.json -t l1

python -m scripts.evilslime.train -a argfiles/evilslime/cartpole-v1.json -t noisy
python -m scripts.evilslime.evaluate -a argfiles/evilslime/cartpole-v1.json -t noisy

python -m scripts.evilslime.train -a argfiles/evilslime/acrobot-v1.json -t noisy
python -m scripts.evilslime.evaluate -a argfiles/evilslime/acrobot-v1.json -t noisy

python -m scripts.evilslime.train -a argfiles/evilslime/pendulum-v1.json -t noisy
python -m scripts.evilslime.evaluate -a argfiles/evilslime/pendulum-v1.json -t noisy

python -m scripts.evilslime.train -a argfiles/evilslime/lunarlander-v2.json -t noisy
python -m scripts.evilslime.evaluate -a argfiles/evilslime/lunarlander-v2.json -t noisy

python -m scripts.evilslime.train -a argfiles/evilslime/bipedalwalker-v3.json -t noisy
python -m scripts.evilslime.evaluate -a argfiles/evilslime/bipedalwalker-v3.json -t noisy

python -m scripts.blockland.train -a argfiles/blockland/twosides-rw.json -t base
python -m scripts.blockland.evaluate -a argfiles/blockland/twosides-rw.json -t base

python -m scripts.blockland.train -a argfiles/blockland/twosides-nw.json -t base
python -m scripts.blockland.evaluate -a argfiles/blockland/twosides-nw.json -t base

python -m scripts.blockland.train -a argfiles/blockland/junction-nw.json -t base
python -m scripts.blockland.evaluate -a argfiles/blockland/junction-nw.json -t base
