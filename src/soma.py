from mpmath import mp, fsum, mpf

# Definir a precisão desejada
mp.dps = 50

# Lista de valores flutuantes
valores = [
    mpf('0.008245983273868913049701497842146357569621317450630166'),
    mpf('0.025773299772891788624068067186999263228590295216130713'),
    mpf('0.0055201723020395636126102920080755805047435038272588774'),
    mpf('0.073548977498818220196432451513587166443565125534085588'),
    mpf('0.0099478672686512107065549496587946883835090758550766848'),
    mpf('0.0091204625551532336346486126079370807250319593803877306'),
    mpf('0.074579121948684403642770359133590123105140860661684608'),
    mpf('0.05095352705603799156241080841629986474074278993240345'),
    mpf('0.049341519586462062350336246602118091864151948983316444'),
    mpf('0.016984335504433122754301062853513399633634885386610066')
]

# Realizar a soma
soma = fsum(valores)

# Imprimir o resultado
print(soma)