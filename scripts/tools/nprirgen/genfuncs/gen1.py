import numpy as np
import funcs


# Generate RIRs. 
def genrir1(soundVelocity, fs, receiverPositions, sourcePosition, roomMeasures, betaCoeffs, orientation, isHighPassFilter, nOrder, nSamples, micType):
    nMicrophones = len(receiverPositions)

    # Check if the reflection order is valid. 
    if nOrder < -1:
        raise ValueError('Reflection order must be an integer >= -1.')

    W = 2 * np.pi * 100 / fs  # cut-off freq (100 Hz)
    R1 = np.exp(-W)
    B1 = 2 * R1 * np.cos(W)
    B2 = -R1 * R1
    A1 = -(1 + R1)

    # tmp vars related to RIR generation
    Fc = 1  # The cut-off frequency equals fs/2 - Fc is the normalized cut-off frequency.
    Tw = 2 * funcs.symceil(0.004 * fs)  # The width of the low-pass FIR equals 8 ms
    cTs = soundVelocity / fs;

    Rm = np.zeros(3)
    Rp_plus_Rm = np.zeros(3)
    refl = np.zeros(3)
    LPI = np.zeros(Tw)

    imp = np.zeros((nMicrophones, nSamples))

    s = sourcePosition / cTs
    L = roomMeasures / cTs

    for idxMicrophone in range(nMicrophones):
        r = receiverPositions[idxMicrophone] / cTs
        n = np.ceil(nSamples / (2 * L)).astype(int)

        for mx in range(-n[0], n[0]+1):
            Rm[0] = 2 * mx * L[0]

            for my in range(-n[1], n[1]+1):
                Rm[1] = 2 * my * L[1]

                for mz in range(-n[2], n[2]+1):
                    Rm[2] = 2 * mz * L[2]

                    for q in range(0, 2):
                        Rp_plus_Rm[0] = (1 - 2 * q) * s[0] - r[0] + Rm[0]
                        refl[0] = np.power(betaCoeffs[0], np.abs(mx - q)) * np.power(betaCoeffs[1], np.abs(mx))

                        for j in range(0, 2):
                            Rp_plus_Rm[1] = (1 - 2 * j) * s[1] - r[1] + Rm[1]
                            refl[1] = np.power(betaCoeffs[2], np.abs(my - j)) * np.power(betaCoeffs[3], np.abs(my))

                            for k in range(0, 2):
                                Rp_plus_Rm[2] = (1 - 2 * k) * s[2] - r[2] + Rm[2]
                                refl[2] = np.power(betaCoeffs[4], np.abs(mz - k)) * np.power(betaCoeffs[5], np.abs(mz))

                                dist = np.sqrt(Rp_plus_Rm[0]**2 + Rp_plus_Rm[1]**2 + Rp_plus_Rm[2]**2)

                                if np.abs(2*mx-q) + np.abs(2*my-j) + np.abs(2*mz-k) <= nOrder or nOrder == -1:
                                    fdist = np.floor(dist)

                                    if fdist < nSamples:
                                        gain = funcs.sim_microphone(Rp_plus_Rm[0], Rp_plus_Rm[1], Rp_plus_Rm[2], orientation, micType) * refl[0] * refl[1] * refl[2] / (4 * np.pi * dist * cTs)

                                        for i in range(Tw):
                                            LPI[i] =  0.5 * (1 - np.cos(2 * np.pi * ((i+1-(dist-fdist))/Tw))) * Fc * np.sinc(Fc * (i+1-(dist-fdist)-(Tw/2)))
                                            
                                        startPosition = int(fdist - (Tw/2) + 1)
                                        for i in range(Tw):
                                            if startPosition + i >= 0 and startPosition + i < nSamples:
                                                imp[idxMicrophone, startPosition + i] += gain * LPI[i]


		# 'Original' high-pass filter as proposed by Allen and Berkley.
        if isHighPassFilter:
            Y = np.zeros(3)
                
            for idx in range(nSamples):
                X0 = imp[idxMicrophone, idx]
                Y[2] = Y[1]
                Y[1] = Y[0]
                Y[0] = B1 * Y[1] + B2 * Y[2] + X0
                imp[idxMicrophone, idx] = Y[0] + A1 * Y[1] + R1 * Y[2]

    return imp