import matplotlib.pyplot as plot


def draw_b256_l101():
    data = {}
    data[5] = 0.042
    data[6] = 0.047
    data[7] = 0.054
    data[8] = 0.053
    data[9] = 0.139

    data[10] = 0.044
    data[11] = 0.069
    data[12] = 0.04
    data[13] = 0.183
    data[15] = 0.097
    data[18] = 0.118
    data[17] = 0.068

    data[20] = 0.077
    data[23] = 0.17
    data[25] = 0.219
    data[27] = 0.14
    data[28] = 0.16
    data[29] = 0.278

    data[30] = 0.19
    data[31] = 0.171
    data[32] = 0.18
    data[33] = 0.254
    data[34] = 0.212
    data[35] = 0.247
    data[38] = 0.276
    data[39] = 0.261

    data[40] = 0.188
    data[41] = 0.262
    data[42] = 0.2
    data[43] = 0.22
    data[45] = 0.214
    data[47] = 0.272
    data[48] = 0.176
    data[49] = 0.205

    data[50] = 0.276
    data[51] = 0.232
    data[53] = 0.238
    data[55] = 0.251
    data[58] = 0.256

    data[60] = 0.242
    data[63] = 0.186
    data[64] = 0.197
    data[65] = 0.267
    data[66] = 0.265

    data[70] = 0.216
    data[75] = 0.243
    data[78] = 0.21

    data[80] = 0.196
    data[81] = 0.247
    data[83] = 0.214
    data[85] = 0.265
    data[88] = 0.223

    data[90] = 0.257
    data[93] = 0.288
    data[95] = 0.288

    data[100] = 0.269
    data[105] = 0.305
    data[110] = 0.275
    data[115] = 0.311
    data[120] = 0.278  # уже нет макс депс
    data[125] = 0.31
    data[130] = 0.352   # уже нет макс депс
    data[140] = 0.279
    data[150] = 0.323
    data[160] = 0.366
    data[170] = 0.337
    data[180] = 0.308
    data[190] = 0.371  # уже нет макс депс

    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".")
    plot.xlabel('max depth')
    plot.ylabel('f1')
    plot.title('buckets = 256, leaf = 101')
    plot.xticks([i for i in range(0, 200, 10)])
    plot.grid()
    plot.show()


def draw_d130_l101():
    data = {}
    data[3] = 0.246
    data[4] = 0.297
    data[5] = 0.34
    data[6] = 0.345
    data[7] = 0.383
    data[8] = 0.316
    data[9] = 0.309

    data[10] = 0.347
    data[11] = 0.378
    data[12] = 0.297
    data[13] = 0.343
    data[14] = 0.351
    data[15] = 0.3
    data[16] = 0.359
    data[17] = 0.358
    data[18] = 0.362
    data[19] = 0.318

    data[20] = 0.349
    data[21] = 0.327
    data[22] = 0.324
    data[23] = 0.286
    data[24] = 0.314
    data[25] = 0.359
    data[26] = 0.302
    data[27] = 0.311
    data[28] = 0.341
    data[29] = 0.35

    data[30] = 0.332
    data[35] = 0.313

    data[40] = 0.296
    data[45] = 0.31

    data[50] = 0.327
    data[55] = 0.314

    data[60] = 0.349
    data[70] = 0.352
    data[75] = 0.334
    data[78] = 0.351
    data[79] = 0.37
    data[80] = 0.404
    data[83] = 0.3
    data[90] = 0.389

    data[100] = 0.348
    data[110] = 0.313
    data[115] = 0.353
    data[130] = 0.35
    data[140] = 0.374
    data[145] = 0.327
    data[150] = 0.362
    data[155] = 0.393

    data[160] = 0.377
    data[164] = 0.3
    data[165] = 0.41
    data[166] = 0.437
    data[167] = 0.397
    data[168] = 0.398
    data[170] = 0.397
    data[175] = 0.285
    data[190] = 0.324
    data[205] = 0.341
    data[220] = 0.343
    data[236] = 0.384
    data[238] = 0.383
    data[240] = 0.379
    data[260] = 0.322

    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".")
    plot.xlabel('bucket size')
    plot.ylabel('f1')
    plot.title('max depth = 130, leaf = 101')
    plot.xticks([i for i in range(0, 270, 10)])
    plot.grid()
    plot.show()


def dt_with_wo():
    data = {}
    # data[2] = (0.377 + 0.414 + 0.377 + 0.367) / 4
    # data[3] = (0.384 + 0.455 + 0.42 + 0.425) / 4
    # data[4] = (0.415 + 0.464 + 0.441 + 0.428) / 4
    # data[5] = (0.389 + 0.432 + 0.401 + 0.402) / 4
    # data[6] = (0.42 + 0.389 + 0.425 + 0.404) / 4
    # data[7] = (0.407 + 0.413 + 0.441 + 0.409) / 4
    # data[8] = (0.42 + 0.449 + 0.401 + 0.389) / 4
    # data[9] = (0.437 + 0.426 + 0.439 + 0.403) / 4
    # data[10] = (0.452 + 0.453 + 0.402 + 0.426) / 4
    # data[11] = (0.427 + 0.433 + 0.411 + 0.426) / 4
    # data[12] = (0.422 + 0.39 + 0.423 + 0.426) / 4
    # data[13] = (0.366 + 0.37 + 0.407 + 0.398) / 4
    # data[14] = (0.428 + 0.417 + 0.358 + 0.42) / 4
    # data[15] = (0.414 + 0.359 + 0.412 + 0.406) / 4
    # data[16] = (0.42 + 0.395 + 0.373 + 0.423) / 4
    # data[17] = (0.395 + 0.477 + 0.386 + 0.415) / 4
    # data[18] = (0.374 + 0.404 + 0.375 + 0.429) / 4
    # data[19] = (0.396 + 0.387 + 0.399 + 0.359) / 4
    # data[20] = (0.426 + 0.413 + 0.406 + 0.432) / 4
    # data[21] = (0.445 + 0.401 + 0.377 + 0.394) / 4
    data[2] = (0.421 + 0.377 + 0.425) / 3
    data[3] = (0.436 + 0.424 + 0.432) / 3
    data[4] = (0.441 + 0.433 + 0.446) / 3
    data[5] = (0.442 + 0.449 + 0.432) / 3
    data[6] = (0.462 + 0.462 + 0.472) / 3
    data[7] = (0.448 + 0.463 + 0.468) / 3
    data[8] = (0.448 + 0.452 + 0.472) / 3
    data[9] = (0.444 + 0.45 + 0.441) / 3
    data[10] = (0.453 + 0.45 + 0.452) / 3
    data[11] = (0.451 + 0.465 + 0.45) / 3
    data[12] = (0.463 + 0.455 + 0.463) / 3
    data[13] = (0.462 + 0.469 + 0.451) / 3
    data[14] = (0.462 + 0.463 + 0.46) / 3
    data[15] = (0.466 + 0.463 + 0.448) / 3
    data[16] = (0.457 + 0.466 + 0.449) / 3
    data[17] = (0.467 + 0.448 + 0.469) / 3
    data[18] = (0.454 + 0.458 + 0.445) / 3
    data[19] = (0.462 + 0.471 + 0.469) / 3
    data[20] = (0.454 + 0.474 + 0.46) / 3
    data[21] = (0.465 + 0.463 + 0.447) / 3

    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
              label='with features', linewidth=1)

    plot.xlabel('max depth')
    plot.ylabel('f1')
    plot.title('leaf size = 431, num of buckets = 8')
    plot.xticks([i for i in range(2, 22, 2)])
    plot.ylim(0.3, 0.55)
    plot.grid()

    data = {}
    # data[2] = (0.399 + 0.401 + 0.368 + 0.417) / 4
    # data[3] = (0.43 + 0.382 + 0.394 + 0.399) / 4
    # data[4] = (0.378 + 0.466 + 0.464 + 0.41) / 4
    # data[5] = (0.413 + 0.456 + 0.429 + 0.455) / 4
    # data[6] = (0.437 + 0.447 + 0.381 + 0.391) / 4
    # data[7] = (0.397 + 0.417 + 0.396 + 0.43) / 4
    # data[8] = (0.41 + 0.444 + 0.446 + 0.411) / 4
    # data[9] = (0.446 + 0.355 + 0.419 + 0.457) / 4
    # data[10] = (0.428 + 0.407 + 0.426 + 0.412) / 4
    # data[11] = (0.366 + 0.422 + 0.434 + 0.414) / 4
    # data[12] = (0.371 + 0.398 + 0.399 + 0.374) / 4
    # data[13] = (0.402 + 0.421 + 0.374 + 0.419) / 4
    # data[14] = (0.385 + 0.389 + 0.444 + 0.402) / 4
    # data[15] = (0.424 + 0.436 + 0.402 + 0.375) / 4
    # data[16] = (0.406 + 0.409 + 0.427 + 0.42) / 4
    # data[17] = (0.381 + 0.397 + 0.378 + 0.405) / 4
    # data[18] = (0.401 + 0.381 + 0.391 + 0.406) / 4
    # data[19] = (0.361 + 0.424 + 0.419 + 0.414) / 4
    # data[20] = (0.398 + 0.439 + 0.454 + 0.407) / 4
    # data[21] = (0.423 + 0.437 + 0.375 + 0.396) / 4
    data[2] = (0.421 + 0.4 + 0.378) / 3
    data[3] = (0.44 + 0.426 + 0.436) / 3
    data[4] = (0.439 + 0.426 + 0.429) / 3
    data[5] = (0.444 + 0.449 + 0.441) / 3
    data[6] = (0.461 + 0.453 + 0.463) / 3
    data[7] = (0.466 + 0.465 + 0.468) / 3
    data[8] = (0.471 + 0.473 + 0.457) / 3
    data[9] = (0.455 + 0.458 + 0.465) / 3
    data[10] = (0.461 + 0.445 + 0.474) / 3
    data[11] = (0.452 + 0.462 + 0.445) / 3
    data[12] = (0.436 + 0.469 + 0.456) / 3
    data[13] = (0.45 + 0.445 + 0.448) / 3
    data[14] = (0.455 + 0.458 + 0.471) / 3
    data[15] = (0.453 + 0.459 + 0.46) / 3
    data[16] = (0.448 + 0.474 + 0.458) / 3
    data[17] = (0.435 + 0.437 + 0.452) / 3
    data[18] = (0.455 + 0.446 + 0.471) / 3
    data[19] = (0.462 + 0.462 + 0.454) / 3
    data[20] = (0.45 + 0.456 + 0.46) / 3
    data[21] = (0.443 + 0.465 + 0.457) / 3

    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
              label='without features', linewidth=1)

    plot.legend()
    plot.show()


def dt_d2_22_4times():
    data = {}
    data[2] = (0.384 + 0.398) / 2
    data[3] = (0.426 + 0.419) / 2
    data[4] = (0.458 + 0.466) / 2
    data[5] = (0.456 + 0.432) / 2
    data[6] = (0.486 + 0.461) / 2
    data[7] = (0.469 + 0.466) / 2
    data[8] = (0.451 + 0.447) / 2
    data[9] = (0.462 + 0.46) / 2
    data[10] = (0.461 + 0.473) / 2
    data[11] = (0.492 + 0.459) / 2
    data[12] = (0.469 + 0.47) / 2
    data[13] = (0.469 + 0.446) / 2
    data[14] = (0.444 + 0.441) / 2
    data[15] = (0.468 + 0.459) / 2
    data[16] = (0.472 + 0.449) / 2
    data[17] = (0.466 + 0.468) / 2
    data[18] = (0.467 + 0.478) / 2
    data[19] = (0.463 + 0.457) / 2
    data[20] = (0.483 + 0.441) / 2
    data[21] = (0.456 + 0.468) / 2
    # data[22] = (0.453)
    # data[23] = (0.472)
    # data[24] = (0.45)
    # data[25] = (0.461)
    # data[26] = (0.457)
    # data[27] = (0.462)
    # data[28] = (0.46)
    # data[29] = (0.483)
    # data[30] = (0.443)
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
              label='number of buckets: 4', linewidth=1)

    data = {}
    data[2] = (0.396 + 0.425) / 2
    data[3] = (0.439 + 0.437) / 2
    data[4] = (0.434 + 0.436) / 2
    data[5] = (0.447 + 0.455) / 2
    data[6] = (0.471 + 0.447) / 2
    data[7] = (0.467 + 0.454) / 2
    data[8] = (0.482 + 0.462) / 2
    data[9] = (0.456 + 0.462) / 2
    data[10] = (0.439 + 0.462) / 2
    data[11] = (0.449 + 0.451) / 2
    data[12] = (0.457 + 0.449) / 2
    data[13] = (0.477 + 0.446) / 2
    data[14] = (0.468 + 0.453) / 2
    data[15] = (0.466 + 0.446) / 2
    data[16] = (0.465 + 0.472) / 2
    data[17] = (0.46 + 0.447) / 2
    data[18] = (0.479 + 0.458) / 2
    data[19] = (0.46 + 0.455) / 2
    data[20] = (0.49 + 0.461) / 2
    data[21] = (0.471 + 0.47) / 2
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
              label='number of buckets: 8', linewidth=1)

    data = {}
    data[2] = (0.407 + 0.424) / 2
    data[3] = (0.45 + 0.437) / 2
    data[4] = (0.457 + 0.457) / 2
    data[5] = (0.439 + 0.456) / 2
    data[6] = (0.466 + 0.454) / 2
    data[7] = (0.467 + 0.451) / 2
    data[8] = (0.476 + 0.462) / 2
    data[9] = (0.46 + 0.483) / 2
    data[10] = (0.453 + 0.478) / 2
    data[11] = (0.495 + 0.479) / 2
    data[12] = (0.469 + 0.461) / 2
    data[13] = (0.447 + 0.453) / 2
    data[14] = (0.442 + 0.464) / 2
    data[15] = (0.453 + 0.443) / 2
    data[16] = (0.457 + 0.447) / 2
    data[17] = (0.454 + 0.463) / 2
    data[18] = (0.454 + 0.448) / 2
    data[19] = (0.448 + 0.457) / 2
    data[20] = (0.481 + 0.455) / 2
    data[21] = (0.466 + 0.459) / 2
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
              label='number of buckets: 16', linewidth=1)

    data = {}
    data[2] = (0.417 + 0.413) / 2
    data[3] = (0.442 + 0.437) / 2
    data[4] = (0.426 + 0.446) / 2
    data[5] = (0.447 + 0.465) / 2
    data[6] = (0.465 + 0.465) / 2
    data[7] = (0.464 + 0.474) / 2
    data[8] = (0.451 + 0.454) / 2
    data[9] = (0.468 + 0.443) / 2
    data[10] = (0.474 + 0.461) / 2
    data[11] = (0.454 + 0.471) / 2
    data[12] = (0.462 + 0.45) / 2
    data[13] = (0.463 + 0.462) / 2
    data[14] = (0.47 + 0.453) / 2
    data[15] = (0.458 + 0.463) / 2
    data[16] = (0.458 + 0.46) / 2
    data[17] = (0.454 + 0.465) / 2
    data[18] = (0.447 + 0.47) / 2
    data[19] = (0.464 + 0.45) / 2
    data[20] = (0.469 + 0.464) / 2
    data[21] = (0.458 + 0.466) / 2
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
              label='number of buckets: 32', linewidth=1)

    data = {}
    data[2] = (0.405 + 0.411) / 2
    data[3] = (0.451 + 0.437) / 2
    data[4] = (0.448 + 0.459) / 2
    data[5] = (0.449 + 0.45) / 2
    data[6] = (0.456 + 0.476) / 2
    data[7] = (0.474 + 0.445) / 2
    data[8] = (0.465 + 0.448) / 2
    data[9] = (0.458 + 0.469) / 2
    data[10] = (0.444 + 0.464) / 2
    data[11] = (0.464 + 0.461) / 2
    data[12] = (0.463 + 0.445) / 2
    data[13] = (0.45 + 0.461) / 2
    data[14] = (0.471 + 0.457) / 2
    data[15] = (0.476 + 0.468) / 2
    data[16] = (0.442 + 0.455) / 2
    data[17] = (0.452 + 0.459) / 2
    data[18] = (0.472 + 0.473) / 2
    data[19] = (0.444 + 0.442) / 2
    data[20] = (0.474 + 0.467) / 2
    data[21] = (0.447 + 0.442) / 2
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
              label='number of buckets: 64', linewidth=1)

    plot.xlabel('max depth')
    plot.ylabel('f1')
    plot.title('leaf size = 431')
    plot.xticks([i for i in range(2, 22, 2)])
    plot.ylim(0.35, 0.55)
    plot.grid()
    plot.legend()
    plot.show()


def draw_b4_64__l89():
    data = {}
    data[2] = (0.377 + 0.414 + 0.377 + 0.367) / 4
    data[3] = (0.384 + 0.455 + 0.42 + 0.425) / 4
    data[4] = (0.415 + 0.464 + 0.441 + 0.428) / 4
    data[5] = (0.389 + 0.432 + 0.401 + 0.402) / 4
    data[6] = (0.42 + 0.389 + 0.425 + 0.404) / 4
    data[7] = (0.407 + 0.413 + 0.441 + 0.409) / 4
    data[8] = (0.42 + 0.449 + 0.401 + 0.389) / 4
    data[9] = (0.437 + 0.426 + 0.439 + 0.403) / 4
    data[10] = (0.452 + 0.453 + 0.402 + 0.426) / 4
    data[11] = (0.427 + 0.433 + 0.411 + 0.426) / 4
    data[12] = (0.422 + 0.39 + 0.423 + 0.426) / 4
    data[13] = (0.366 + 0.37 + 0.407 + 0.398) / 4
    data[14] = (0.428 + 0.417 + 0.358 + 0.42) / 4
    data[15] = (0.414 + 0.359 + 0.412 + 0.406) / 4
    data[16] = (0.42 + 0.395 + 0.373 + 0.423) / 4
    data[17] = (0.395 + 0.477 + 0.386 + 0.415) / 4
    data[18] = (0.374 + 0.404 + 0.375 + 0.429) / 4
    data[19] = (0.396 + 0.387 + 0.399 + 0.359) / 4
    data[20] = (0.426 + 0.413 + 0.406 + 0.432) / 4
    data[21] = (0.445 + 0.401 + 0.377 + 0.394) / 4
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".", label='b=4', linewidth=1)

    plot.xlabel('max depth')
    plot.ylabel('f1')
    plot.title('leaf size = 89')
    plot.xticks([i for i in range(2, 22, 2)])
    plot.ylim(0.3, 0.6)
    plot.grid()

    data = {}
    data[2] = (0.398 + 0.389 + 0.36 + 0.423) / 4
    data[3] = (0.461 + 0.42 + 0.387 + 0.444) / 4
    data[4] = (0.432 + 0.443 + 0.417 + 0.467) / 4
    data[5] = (0.416 + 0.455 + 0.437 + 0.419) / 4
    data[6] = (0.363 + 0.369 + 0.444 + 0.433) / 4
    data[7] = (0.417 + 0.429 + 0.441 + 0.456) / 4
    data[8] = (0.384 + 0.424 + 0.354 + 0.382) / 4
    data[9] = (0.37 + 0.378 + 0.425 + 0.439) / 4
    data[10] = (0.412 + 0.406 + 0.452 + 0.429) / 4
    data[11] = (0.408 + 0.376 + 0.422 + 0.407) / 4
    data[12] = (0.415 + 0.397 + 0.429 + 0.383) / 4
    data[13] = (0.4 + 0.386 + 0.375 + 0.389) / 4
    data[14] = (0.381 + 0.424 + 0.374 + 0.45) / 4
    data[15] = (0.416 + 0.392 + 0.428 + 0.357) / 4
    data[16] = (0.388 + 0.43 + 0.423 + 0.399) / 4
    data[17] = (0.441 + 0.435 + 0.387 + 0.422) / 4
    data[18] = (0.46 + 0.394 + 0.386 + 0.431) / 4
    data[19] = (0.399 + 0.43 + 0.373 + 0.416) / 4
    data[20] = (0.416 + 0.367 + 0.39 + 0.426) / 4
    data[21] = (0.391 + 0.397 + 0.388 + 0.393) / 4
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".", label='b=8', linewidth=1)

    data = {}
    data[2] = (0.384 + 0.415 + 0.405 + 0.408) / 4
    data[3] = (0.449 + 0.425 + 0.41 + 0.433) / 4
    data[4] = (0.423 + 0.451 + 0.41 + 0.47) / 4
    data[5] = (0.423 + 0.41 + 0.465 + 0.446) / 4
    data[6] = (0.413 + 0.451 + 0.401 + 0.397) / 4
    data[7] = (0.387 + 0.403 + 0.396 + 0.412) / 4
    data[8] = (0.427 + 0.438 + 0.373 + 0.448) / 4
    data[9] = (0.434 + 0.402 + 0.385 + 0.426) / 4
    data[10] = (0.424 + 0.427 + 0.441 + 0.429) / 4
    data[11] = (0.427 + 0.408 + 0.386 + 0.33) / 4
    data[12] = (0.407 + 0.406 + 0.396 + 0.398) / 4
    data[13] = (0.407 + 0.447 + 0.371 + 0.373) / 4
    data[14] = (0.44 + 0.393 + 0.439 + 0.392) / 4
    data[15] = (0.421 + 0.409 + 0.401 + 0.353) / 4
    data[16] = (0.444 + 0.405 + 0.403 + 0.417) / 4
    data[17] = (0.417 + 0.385 + 0.381 + 0.418) / 4
    data[18] = (0.434 + 0.382 + 0.429 + 0.38) / 4
    data[19] = (0.407 + 0.362 + 0.394 + 0.449) / 4
    data[20] = (0.413 + 0.407 + 0.412 + 0.409) / 4
    data[21] = (0.416 + 0.454 + 0.415 + 0.432) / 4
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".", label='b=16', linewidth=1)

    data[2] = (0.366 + 0.383 + 0.414 + 0.372) / 4
    data[3] = (0.426 + 0.403 + 0.405 + 0.422) / 4
    data[4] = (0.481 + 0.442 + 0.431 + 0.426) / 4
    data[5] = (0.422 + 0.429 + 0.413 + 0.459) / 4
    data[6] = (0.451 + 0.422 + 0.448 + 0.413) / 4
    data[7] = (0.438 + 0.415 + 0.441 + 0.429) / 4
    data[8] = (0.431 + 0.451 + 0.427 + 0.428) / 4
    data[9] = (0.412 + 0.391 + 0.442 + 0.371) / 4
    data[10] = (0.426 + 0.434 + 0.368 + 0.407) / 4
    data[11] = (0.437 + 0.406 + 0.422 + 0.465) / 4
    data[12] = (0.416 + 0.393 + 0.456 + 0.436) / 4
    data[13] = (0.458 + 0.436 + 0.421 + 0.418) / 4
    data[14] = (0.32 + 0.418 + 0.422 + 0.368) / 4
    data[15] = (0.402 + 0.406 + 0.357 + 0.392) / 4
    data[16] = (0.385 + 0.413 + 0.421 + 0.41) / 4
    data[17] = (0.373 + 0.399 + 0.42 + 0.405) / 4
    data[18] = (0.422 + 0.466 + 0.386 + 0.413) / 4
    data[19] = (0.417 + 0.397 + 0.401 + 0.393) / 4
    data[20] = (0.425 + 0.406 + 0.388 + 0.388) / 4
    data[21] = (0.422 + 0.37 + 0.398 + 0.39) / 4
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".", label='b=32', linewidth=1)

    data[2] = (0.389 + 0.397 + 0.424 + 0.366) / 4
    data[3] = (0.438 + 0.414 + 0.434 + 0.424) / 4
    data[4] = (0.468 + 0.458 + 0.43 + 0.451) / 4
    data[5] = (0.412 + 0.432 + 0.408 + 0.372) / 4
    data[6] = (0.416 + 0.402 + 0.408 + 0.451) / 4
    data[7] = (0.399 + 0.433 + 0.451 + 0.435) / 4
    data[8] = (0.415 + 0.419 + 0.414 + 0.408) / 4
    data[9] = (0.442 + 0.433 + 0.392 + 0.401) / 4
    data[10] = (0.418 + 0.443 + 0.413 + 0.446) / 4
    data[11] = (0.427 + 0.414 + 0.378 + 0.405) / 4
    data[12] = (0.425 + 0.38 + 0.442 + 0.393) / 4
    data[13] = (0.413 + 0.4 + 0.384 + 0.389) / 4
    data[14] = (0.395 + 0.429 + 0.395 + 0.428) / 4
    data[15] = (0.401 + 0.426 + 0.435 + 0.41) / 4
    data[16] = (0.389 + 0.424 + 0.408 + 0.391) / 4
    data[17] = (0.367 + 0.413 + 0.415 + 0.41) / 4
    data[18] = (0.417 + 0.419 + 0.392 + 0.41) / 4
    data[19] = (0.395 + 0.426 + 0.424 + 0.386) / 4
    data[20] = (0.416 + 0.375 + 0.396 + 0.434) / 4
    data[21] = (0.384 + 0.422 + 0.333 + 0.384) / 4
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".", label='b=64', linewidth=1)

    plot.legend()
    plot.show()


# def draw_d5_b6():
#     data = {}
#     data[3] = 0.452
#     data[5] = 0.453
#     data[7] = 0.448
#     data[9] = 0.465
#     data[11] = 0.449
#     data[21] = 0.407
#     data[31] = 0.446
#     data[41] = 0.444
#     data[51] = 0.427
#     data[61] = 0.394
#     data[71] = 0.411
#     data[81] = 0.433
#     data[91] = 0.424
#     data[101] = 0.436
#     data[111] = 0.414
#     data[121] = 0.446
#     data[131] = 0.434
#     data[141] = 0.462
#     data[151] = 0.445
#     data[161] = 0.428
#     data[171] = 0.428
#     data[181] = 0.438
#     data[191] = 0.397
#     data[201] = 0.471
#     data[251] = 0.445
#     data[301] = 0.449
#     data[351] = 0.455
#     data[401] = 0.416
#     data[451] = 0.463
#     data[501] = 0.422
#     data[551] = 0.406
#     data[601] = 0.426
#     data[651] = 0.427
#
#     plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".", linewidth=1, label='b=6')
#     plot.xlabel('leaf size')
#     plot.ylabel('f1')
#     plot.title('max depth = 130, leaf = 101')
#     plot.xticks([i for i in range(0, 660, 50)])
#     plot.grid()
#
#     data = {}
#     data[51] = 0.415
#     data[101] = 0.443
#     data[151] = 0.412
#     data[201] = 0.399
#     data[251] = 0.417
#     data[301] = 0.419
#     data[351] = 0.418
#
#     plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".", linewidth=1, label='b=4')
#
#     plot.legend()
#     plot.show()


def draw_leaf_dep():
    data = {}
    data[11] = 0.4222904272730881
    data[21] = 0.4189259082161812
    data[31] = 0.42945146512270915
    data[41] = 0.4061791020233892
    data[51] = 0.4235927662710874
    data[61] = 0.44214808097538416
    data[71] = 0.43360543647229133
    data[81] = 0.4232407719124133
    data[91] = 0.4314816203231003

    data[101] = 0.4420399614612254
    data[111] = 0.42959575073663203
    data[121] = 0.4277748925456805
    data[131] = 0.4511294677750335
    data[141] = 0.4396700791000167
    data[151] = 0.4379808990211724
    data[161] = 0.4257870799298446
    data[171] = 0.42251936102068
    data[181] = 0.4128696992069085
    data[191] = 0.423597114372236

    data[201] = 0.4448257400079944
    data[211] = 0.4217797342413893
    data[221] = 0.4300187552369149
    data[231] = 0.40822573698316333
    data[241] = 0.44592828715300903
    data[251] = 0.4290156405616859
    data[261] = 0.4535204099572435
    data[271] = 0.42914514398700515
    data[281] = 0.414957306177291
    data[291] = 0.42234534560113207

    data[301] = 0.420254758759255
    data[311] = 0.41854895317228236
    data[321] = 0.43608396331033067
    data[331] = 0.41784397121806116
    data[341] = 0.437522677282655
    data[351] = 0.42466425609716973
    data[361] = 0.4433004987649143
    data[371] = 0.4500358789983773
    data[381] = 0.43741293641939893
    data[391] = 0.4183922919354891

    data[401] = 0.44009982702858935
    data[411] = 0.43749255901379946
    data[421] = 0.40971075871782825
    data[431] = 0.45868439769883196
    data[441] = 0.43281999786019254
    data[451] = 0.4143821558088423
    data[461] = 0.4410044943841792
    data[471] = 0.4402138997153088
    data[481] = 0.441543863535302
    data[491] = 0.4323107004661666

    data[501] = 0.41793596375623276
    data[511] = 0.41884739171754637
    data[521] = 0.42748952944830854
    data[531] = 0.4099015850212868
    data[541] = 0.4160381714853393
    data[551] = 0.4382396473784593
    data[561] = 0.4343088701958571
    data[571] = 0.4341104023250335
    data[581] = 0.40818251253429827
    data[591] = 0.4146239575492468

    data[601] = 0.43706995869229903
    data[611] = 0.44357332949263784
    data[621] = 0.431103607089654
    data[631] = 0.416632188628496
    data[641] = 0.4196245699091979
    data[651] = 0.43389201562441365
    data[661] = 0.45087569682386675
    data[671] = 0.4356217082549105
    data[681] = 0.40389300924508936
    data[691] = 0.4254948066116473
    data[701] = 0.43329026206310567
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".", linewidth=1)

    plot.xlabel('leaf size')
    plot.ylabel('f1')
    # plot.title('num of buckets = 89')
    plot.xticks([i for i in range(31, 711, 50)])
    plot.grid()

    plot.legend()
    plot.show()


def bayes_with():
    data = {}
    data[4] = (0.422 + 0.42) / 2
    data[8] = (0.429 + 0.401) / 2
    data[16] = (0.399 + 0.4) / 2
    data[24] = 0.39
    data[32] = (0.394 + 0.392) / 2
    data[48] = 0.4
    data[64] = (0.391 + 0.395) / 2
    data[96] = 0.404
    data[128] = 0.391
    # data[144] = 0.391
    # data[30] = 0.384
    # data[50] = 0.363
    # data[70] = 0.378
    # data[100] = 0.376
    # data[130] = 0.36
    # data[150] = 0.353
    # data[170] = 0.37
    # data[200] = 0.371
    # data[230] = 0.365
    # data[250] = 0.362
    # data[270] = 0.356
    # data[300] = 0.379
    # data[330] = 0.363
    # data[350] = 0.36
    # data[370] = 0.376
    # data[400] = 0.37
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".", label='with features', linewidth=1)

    plot.xlabel('number of buckets')
    plot.ylabel('f1')
    # plot.title('leaf size = 89')
    plot.xticks([i for i in range(0, 131, 10)])
    # plot.ylim(0.2, 0.6)
    plot.grid()

    data = {}
    data[4] = (0.419 + 0.409) / 2
    data[8] = (0.424 + 0.409) / 2
    data[16] = (0.407 + 0.399) / 2
    data[24] = 0.381
    data[32] = (0.389 + 0.394) / 2
    data[48] = 0.397
    data[64] = (0.395 + 0.382) / 2
    data[96] = 0.38
    data[128] = 0.383
    # data[144] = 0.393
    # data[30] = 0.377
    # data[50] = 0.365
    # data[70] = 0.357
    # data[100] = 0.377
    # data[130] = 0.363
    # data[150] = 0.367
    # data[170] = 0.353
    # data[200] = 0.376
    # data[230] = 0.37
    # data[250] = 0.362
    # data[270] = 0.355
    # data[300] = 0.37
    # data[330] = 0.356
    # data[350] = 0.355
    # data[370] = 0.379
    # data[400] = 0.375
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".", label='without features', linewidth=1)

    plot.legend()
    plot.show()


def rf_d_b():
    data = {}
    # data[4] = (0.443 + 0.451) / 2
    # data[6] = (0.417 + 0.427) / 2
    # data[8] = (0.381 + 0.43) / 2
    # data[10] = (0.447 + 0.446) / 2
    # data[12] = (0.468 + 0.405) / 2
    # data[14] = (0.431 + 0.382) / 2
    # data[16] = (0.467 + 0.427) / 2
    data[3] = (0.443 + 0.417 + 0.381 + 0.382 + 0.435 + 0.4 + 0.385 + 0.462 + 0.416 + 0.428) / 10
    data[4] = (0.392 + 0.432 + 0.446 + 0.438 + 0.46 + 0.422 + 0.383 + 0.413 + 0.444 + 0.445) / 10
    data[5] = (0.425 + 0.395 + 0.445 + 0.431 + 0.413 + 0.436 + 0.441 + 0.409 + 0.454 + 0.422) / 10
    data[6] = (0.419 + 0.448 + 0.441 + 0.464 + 0.444 + 0.43 + 0.46 + 0.442 + 0.422 + 0.472) / 10
    data[7] = (0.403 + 0.446 + 0.497 + 0.442 + 0.47 + 0.444 + 0.406 + 0.455 + 0.43 + 0.484) / 10
    data[8] = (0.435 + 0.439 + 0.45 + 0.441 + 0.46 + 0.463 + 0.454 + 0.416 + 0.481 + 0.434) / 10
    data[9] = (0.397 + 0.437 + 0.476 + 0.425 + 0.449 + 0.447 + 0.486 + 0.486 + 0.508 + 0.433) / 10
    data[10] = (0.433 + 0.436 + 0.422 + 0.48 + 0.498 + 0.47 + 0.47 + 0.453 + 0.397 + 0.454) / 10
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
              label='number of features = 0.9 * sqrt(n)', linewidth=1)

    plot.xlabel('max depth')
    plot.ylabel('f1')
    plot.title('leaf size = 431')
    plot.xticks([i for i in range(3, 11, 1)])
    # plot.ylim(0.2, 0.6)
    plot.grid()

    data = {}
    data[3] = (0.412 + 0.425 + 0.464 + 0.406 + 0.444 + 0.409) / 6
    data[4] = (0.391 + 0.42 + 0.426 + 0.423 + 0.407 + 0.426) / 6
    data[5] = (0.446 + 0.395 + 0.46 + 0.413 + 0.425 + 0.429) / 6
    data[6] = (0.414 + 0.418 + 0.465 + 0.428 + 0.474 + 0.431) / 6
    data[7] = (0.438 + 0.456 + 0.443 + 0.416 + 0.439 + 0.445) / 6
    data[8] = (0.429 + 0.427 + 0.469 + 0.409 + 0.403 + 0.417) / 6
    data[9] = (0.487 + 0.45 + 0.424 + 0.394 + 0.463 + 0.473) / 6
    data[10] = (0.431 + 0.492 + 0.448 + 0.426 + 0.472 + 0.471) / 6

    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
              label='number of features = 1.0 * sqrt(n)', linewidth=1)

    data = {}
    data[3] = (0.393 + 0.403 + 0.446 + 0.396 + 0.42 + 0.424) / 6
    data[4] = (0.467 + 0.508 + 0.415 + 0.419 + 0.415 + 0.397) / 6
    data[5] = (0.424 + 0.428 + 0.428 + 0.413 + 0.427 + 0.44) / 6
    data[6] = (0.4 + 0.425 + 0.437 + 0.462 + 0.447 + 0.44) / 6
    data[7] = (0.412 + 0.475 + 0.448 + 0.446 + 0.438 + 0.478) / 6
    data[8] = (0.441 + 0.439 + 0.435 + 0.422 + 0.462 + 0.447) / 6
    data[9] = (0.457 + 0.463 + 0.434 + 0.454 + 0.476 + 0.46) / 6
    data[10] = (0.449 + 0.421 + 0.485 + 0.455 + 0.424 + 0.458) / 6
    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
              label='number of features = 1.1 * sqrt(n)', linewidth=1)

    data = {}
    data[3] = (0.399 + 0.419 + 0.45 + 0.433 + 0.439 + 0.404) / 6
    data[4] = (0.41 + 0.439 + 0.382 + 0.444 + 0.381 + 0.406) / 6
    data[5] = (0.413 + 0.485 + 0.43 + 0.467 + 0.405 + 0.453) / 6
    data[6] = (0.462 + 0.451 + 0.464 + 0.425 + 0.464 + 0.393) / 6
    data[7] = (0.401 + 0.441 + 0.475 + 0.41 + 0.475 + 0.455) / 6
    data[8] = (0.393 + 0.418 + 0.436 + 0.388 + 0.447 + 0.473) / 6
    data[9] = (0.436 + 0.467 + 0.422 + 0.457 + 0.458 + 0.368) / 6
    data[10] = (0.454 + 0.442 + 0.472 + 0.486 + 0.458 + 0.434) / 6

    plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
              label='number of features = 1.2 * sqrt(n)', linewidth=1)

    data = {}
    data[3] = (0.416 + 0.459 + 0.434 + 0.436 + 0.381 + 0.417) / 6
    data[4] = (0.465 + 0.417 + 0.434 + 0.453 + 0.451 + 0.413) / 6
    data[5] = (0.396 + 0.468 + 0.432 + 0.432 + 0.38 + 0.406) / 6
    data[6] = (0.399 + 0.428 + 0.48 + 0.392 + 0.437 + 0.429) / 6
    data[7] = (0.406 + 0.433 + 0.452 + 0.42 + 0.425 + 0.477) / 6
    data[8] = (0.41 + 0.461 + 0.424 + 0.414 + 0.462 + 0.451) / 6
    data[9] = (0.434 + 0.473 + 0.45 + 0.435 + 0.426 + 0.442) / 6
    data[10] = (0.458 + 0.461 + 0.374 + 0.435 + 0.434 + 0.466) / 6

    # plot.plot(list(sorted(data.keys())), list(v for k, v in sorted(data.items())), marker=".",
    #           label='number of features = 1.1 * sqrt(n), without', linewidth=1)
    plot.legend()
    plot.show()


def hist():
    # plot.hist(x=[0.447, 0.508, 0.521], bins=3)
    x = [1, 2, 3, 4, 5]
    plot.bar(x, height=[0.219, 0.447, 0.483, 0.508, 0.521])
    plot.xticks(x, ['random classifier', 'naive bayes classifier', 'decision tree', 'random forest', 'catboost'])
    plot.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55])
    plot.ylabel('max f1')
    plot.grid()
    plot.show()
    # plot.legend()


# rf_d_b()
dt_with_wo()
# hist()
# draw_leaf_dep()
# bayes_with()
# draw_b4_64__l89()
# dt_d2_22_4times()