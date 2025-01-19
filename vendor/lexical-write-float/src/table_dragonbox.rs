//! Pre-computed powers for the Dragonbox algorithm.
//!
//! These tables enable fast scaling of the significant digits
//! of a float to the decimal exponent, with minimal rounding
//! errors, in a 128 or 192-bit representation.
//!
//! DO NOT MODIFY: Generated by `lexical-parse-float/etc/lemire_table.py`
//!
//! This adapted from the Rust implementation, based on the fast-float-rust
//! implementation, and is similarly subject to an Apache2.0/MIT license.

#![cfg(not(feature = "compact"))]
#![doc(hidden)]

pub const SMALLEST_F32_POW5: i32 = -31;
pub const LARGEST_F32_POW5: i32 = 46;
pub const N32_POWERS_OF_FIVE: usize = (LARGEST_F32_POW5 - SMALLEST_F32_POW5 + 1) as usize;
pub const SMALLEST_F64_POW5: i32 = -292;
pub const LARGEST_F64_POW5: i32 = 326;
pub const N64_POWERS_OF_FIVE: usize = (LARGEST_F64_POW5 - SMALLEST_F64_POW5 + 1) as usize;

/// Pre-computed powers-of-5 for the dragonbox algorithm for f32.
///
/// This is very similar to the high bits for `DRAGONBOX64_POWERS_OF_FIVE`,
/// with rounding obviously for the bits.
pub const DRAGONBOX32_POWERS_OF_FIVE: [u64; N32_POWERS_OF_FIVE] = [
    0x81ceb32c4b43fcf5, // 5^-31
    0xa2425ff75e14fc32, // 5^-30
    0xcad2f7f5359a3b3f, // 5^-29
    0xfd87b5f28300ca0e, // 5^-28
    0x9e74d1b791e07e49, // 5^-27
    0xc612062576589ddb, // 5^-26
    0xf79687aed3eec552, // 5^-25
    0x9abe14cd44753b53, // 5^-24
    0xc16d9a0095928a28, // 5^-23
    0xf1c90080baf72cb2, // 5^-22
    0x971da05074da7bef, // 5^-21
    0xbce5086492111aeb, // 5^-20
    0xec1e4a7db69561a6, // 5^-19
    0x9392ee8e921d5d08, // 5^-18
    0xb877aa3236a4b44a, // 5^-17
    0xe69594bec44de15c, // 5^-16
    0x901d7cf73ab0acda, // 5^-15
    0xb424dc35095cd810, // 5^-14
    0xe12e13424bb40e14, // 5^-13
    0x8cbccc096f5088cc, // 5^-12
    0xafebff0bcb24aaff, // 5^-11
    0xdbe6fecebdedd5bf, // 5^-10
    0x89705f4136b4a598, // 5^-9
    0xabcc77118461cefd, // 5^-8
    0xd6bf94d5e57a42bd, // 5^-7
    0x8637bd05af6c69b6, // 5^-6
    0xa7c5ac471b478424, // 5^-5
    0xd1b71758e219652c, // 5^-4
    0x83126e978d4fdf3c, // 5^-3
    0xa3d70a3d70a3d70b, // 5^-2
    0xcccccccccccccccd, // 5^-1
    0x8000000000000000, // 5^0
    0xa000000000000000, // 5^1
    0xc800000000000000, // 5^2
    0xfa00000000000000, // 5^3
    0x9c40000000000000, // 5^4
    0xc350000000000000, // 5^5
    0xf424000000000000, // 5^6
    0x9896800000000000, // 5^7
    0xbebc200000000000, // 5^8
    0xee6b280000000000, // 5^9
    0x9502f90000000000, // 5^10
    0xba43b74000000000, // 5^11
    0xe8d4a51000000000, // 5^12
    0x9184e72a00000000, // 5^13
    0xb5e620f480000000, // 5^14
    0xe35fa931a0000000, // 5^15
    0x8e1bc9bf04000000, // 5^16
    0xb1a2bc2ec5000000, // 5^17
    0xde0b6b3a76400000, // 5^18
    0x8ac7230489e80000, // 5^19
    0xad78ebc5ac620000, // 5^20
    0xd8d726b7177a8000, // 5^21
    0x878678326eac9000, // 5^22
    0xa968163f0a57b400, // 5^23
    0xd3c21bcecceda100, // 5^24
    0x84595161401484a0, // 5^25
    0xa56fa5b99019a5c8, // 5^26
    0xcecb8f27f4200f3a, // 5^27
    0x813f3978f8940985, // 5^28
    0xa18f07d736b90be6, // 5^29
    0xc9f2c9cd04674edf, // 5^30
    0xfc6f7c4045812297, // 5^31
    0x9dc5ada82b70b59e, // 5^32
    0xc5371912364ce306, // 5^33
    0xf684df56c3e01bc7, // 5^34
    0x9a130b963a6c115d, // 5^35
    0xc097ce7bc90715b4, // 5^36
    0xf0bdc21abb48db21, // 5^37
    0x96769950b50d88f5, // 5^38
    0xbc143fa4e250eb32, // 5^39
    0xeb194f8e1ae525fe, // 5^40
    0x92efd1b8d0cf37bf, // 5^41
    0xb7abc627050305ae, // 5^42
    0xe596b7b0c643c71a, // 5^43
    0x8f7e32ce7bea5c70, // 5^44
    0xb35dbf821ae4f38c, // 5^45
    0xe0352f62a19e306f, // 5^46
];

/// Pre-computed powers-of-5 for the dragonbox algorithm for f64.
///
/// This is very similar to the Lemire table, however, a few slight
/// differences exist: this goes from `5^-292` to `5^326`, a different
/// range than Lemire which goes from `5^-342 to 5^308`.
pub const DRAGONBOX64_POWERS_OF_FIVE: [(u64, u64); N64_POWERS_OF_FIVE] = [
    (0xff77b1fcbebcdc4f, 0x25e8e89c13bb0f7b), // 5^-292
    (0x9faacf3df73609b1, 0x77b191618c54e9ad), // 5^-291
    (0xc795830d75038c1d, 0xd59df5b9ef6a2418), // 5^-290
    (0xf97ae3d0d2446f25, 0x4b0573286b44ad1e), // 5^-289
    (0x9becce62836ac577, 0x4ee367f9430aec33), // 5^-288
    (0xc2e801fb244576d5, 0x229c41f793cda740), // 5^-287
    (0xf3a20279ed56d48a, 0x6b43527578c11110), // 5^-286
    (0x9845418c345644d6, 0x830a13896b78aaaa), // 5^-285
    (0xbe5691ef416bd60c, 0x23cc986bc656d554), // 5^-284
    (0xedec366b11c6cb8f, 0x2cbfbe86b7ec8aa9), // 5^-283
    (0x94b3a202eb1c3f39, 0x7bf7d71432f3d6aa), // 5^-282
    (0xb9e08a83a5e34f07, 0xdaf5ccd93fb0cc54), // 5^-281
    (0xe858ad248f5c22c9, 0xd1b3400f8f9cff69), // 5^-280
    (0x91376c36d99995be, 0x23100809b9c21fa2), // 5^-279
    (0xb58547448ffffb2d, 0xabd40a0c2832a78b), // 5^-278
    (0xe2e69915b3fff9f9, 0x16c90c8f323f516d), // 5^-277
    (0x8dd01fad907ffc3b, 0xae3da7d97f6792e4), // 5^-276
    (0xb1442798f49ffb4a, 0x99cd11cfdf41779d), // 5^-275
    (0xdd95317f31c7fa1d, 0x40405643d711d584), // 5^-274
    (0x8a7d3eef7f1cfc52, 0x482835ea666b2573), // 5^-273
    (0xad1c8eab5ee43b66, 0xda3243650005eed0), // 5^-272
    (0xd863b256369d4a40, 0x90bed43e40076a83), // 5^-271
    (0x873e4f75e2224e68, 0x5a7744a6e804a292), // 5^-270
    (0xa90de3535aaae202, 0x711515d0a205cb37), // 5^-269
    (0xd3515c2831559a83, 0x0d5a5b44ca873e04), // 5^-268
    (0x8412d9991ed58091, 0xe858790afe9486c3), // 5^-267
    (0xa5178fff668ae0b6, 0x626e974dbe39a873), // 5^-266
    (0xce5d73ff402d98e3, 0xfb0a3d212dc81290), // 5^-265
    (0x80fa687f881c7f8e, 0x7ce66634bc9d0b9a), // 5^-264
    (0xa139029f6a239f72, 0x1c1fffc1ebc44e81), // 5^-263
    (0xc987434744ac874e, 0xa327ffb266b56221), // 5^-262
    (0xfbe9141915d7a922, 0x4bf1ff9f0062baa9), // 5^-261
    (0x9d71ac8fada6c9b5, 0x6f773fc3603db4aa), // 5^-260
    (0xc4ce17b399107c22, 0xcb550fb4384d21d4), // 5^-259
    (0xf6019da07f549b2b, 0x7e2a53a146606a49), // 5^-258
    (0x99c102844f94e0fb, 0x2eda7444cbfc426e), // 5^-257
    (0xc0314325637a1939, 0xfa911155fefb5309), // 5^-256
    (0xf03d93eebc589f88, 0x793555ab7eba27cb), // 5^-255
    (0x96267c7535b763b5, 0x4bc1558b2f3458df), // 5^-254
    (0xbbb01b9283253ca2, 0x9eb1aaedfb016f17), // 5^-253
    (0xea9c227723ee8bcb, 0x465e15a979c1cadd), // 5^-252
    (0x92a1958a7675175f, 0x0bfacd89ec191eca), // 5^-251
    (0xb749faed14125d36, 0xcef980ec671f667c), // 5^-250
    (0xe51c79a85916f484, 0x82b7e12780e7401b), // 5^-249
    (0x8f31cc0937ae58d2, 0xd1b2ecb8b0908811), // 5^-248
    (0xb2fe3f0b8599ef07, 0x861fa7e6dcb4aa16), // 5^-247
    (0xdfbdcece67006ac9, 0x67a791e093e1d49b), // 5^-246
    (0x8bd6a141006042bd, 0xe0c8bb2c5c6d24e1), // 5^-245
    (0xaecc49914078536d, 0x58fae9f773886e19), // 5^-244
    (0xda7f5bf590966848, 0xaf39a475506a899f), // 5^-243
    (0x888f99797a5e012d, 0x6d8406c952429604), // 5^-242
    (0xaab37fd7d8f58178, 0xc8e5087ba6d33b84), // 5^-241
    (0xd5605fcdcf32e1d6, 0xfb1e4a9a90880a65), // 5^-240
    (0x855c3be0a17fcd26, 0x5cf2eea09a550680), // 5^-239
    (0xa6b34ad8c9dfc06f, 0xf42faa48c0ea481f), // 5^-238
    (0xd0601d8efc57b08b, 0xf13b94daf124da27), // 5^-237
    (0x823c12795db6ce57, 0x76c53d08d6b70859), // 5^-236
    (0xa2cb1717b52481ed, 0x54768c4b0c64ca6f), // 5^-235
    (0xcb7ddcdda26da268, 0xa9942f5dcf7dfd0a), // 5^-234
    (0xfe5d54150b090b02, 0xd3f93b35435d7c4d), // 5^-233
    (0x9efa548d26e5a6e1, 0xc47bc5014a1a6db0), // 5^-232
    (0xc6b8e9b0709f109a, 0x359ab6419ca1091c), // 5^-231
    (0xf867241c8cc6d4c0, 0xc30163d203c94b63), // 5^-230
    (0x9b407691d7fc44f8, 0x79e0de63425dcf1e), // 5^-229
    (0xc21094364dfb5636, 0x985915fc12f542e5), // 5^-228
    (0xf294b943e17a2bc4, 0x3e6f5b7b17b2939e), // 5^-227
    (0x979cf3ca6cec5b5a, 0xa705992ceecf9c43), // 5^-226
    (0xbd8430bd08277231, 0x50c6ff782a838354), // 5^-225
    (0xece53cec4a314ebd, 0xa4f8bf5635246429), // 5^-224
    (0x940f4613ae5ed136, 0x871b7795e136be9a), // 5^-223
    (0xb913179899f68584, 0x28e2557b59846e40), // 5^-222
    (0xe757dd7ec07426e5, 0x331aeada2fe589d0), // 5^-221
    (0x9096ea6f3848984f, 0x3ff0d2c85def7622), // 5^-220
    (0xb4bca50b065abe63, 0x0fed077a756b53aa), // 5^-219
    (0xe1ebce4dc7f16dfb, 0xd3e8495912c62895), // 5^-218
    (0x8d3360f09cf6e4bd, 0x64712dd7abbbd95d), // 5^-217
    (0xb080392cc4349dec, 0xbd8d794d96aacfb4), // 5^-216
    (0xdca04777f541c567, 0xecf0d7a0fc5583a1), // 5^-215
    (0x89e42caaf9491b60, 0xf41686c49db57245), // 5^-214
    (0xac5d37d5b79b6239, 0x311c2875c522ced6), // 5^-213
    (0xd77485cb25823ac7, 0x7d633293366b828c), // 5^-212
    (0x86a8d39ef77164bc, 0xae5dff9c02033198), // 5^-211
    (0xa8530886b54dbdeb, 0xd9f57f830283fdfd), // 5^-210
    (0xd267caa862a12d66, 0xd072df63c324fd7c), // 5^-209
    (0x8380dea93da4bc60, 0x4247cb9e59f71e6e), // 5^-208
    (0xa46116538d0deb78, 0x52d9be85f074e609), // 5^-207
    (0xcd795be870516656, 0x67902e276c921f8c), // 5^-206
    (0x806bd9714632dff6, 0x00ba1cd8a3db53b7), // 5^-205
    (0xa086cfcd97bf97f3, 0x80e8a40eccd228a5), // 5^-204
    (0xc8a883c0fdaf7df0, 0x6122cd128006b2ce), // 5^-203
    (0xfad2a4b13d1b5d6c, 0x796b805720085f82), // 5^-202
    (0x9cc3a6eec6311a63, 0xcbe3303674053bb1), // 5^-201
    (0xc3f490aa77bd60fc, 0xbedbfc4411068a9d), // 5^-200
    (0xf4f1b4d515acb93b, 0xee92fb5515482d45), // 5^-199
    (0x991711052d8bf3c5, 0x751bdd152d4d1c4b), // 5^-198
    (0xbf5cd54678eef0b6, 0xd262d45a78a0635e), // 5^-197
    (0xef340a98172aace4, 0x86fb897116c87c35), // 5^-196
    (0x9580869f0e7aac0e, 0xd45d35e6ae3d4da1), // 5^-195
    (0xbae0a846d2195712, 0x8974836059cca10a), // 5^-194
    (0xe998d258869facd7, 0x2bd1a438703fc94c), // 5^-193
    (0x91ff83775423cc06, 0x7b6306a34627ddd0), // 5^-192
    (0xb67f6455292cbf08, 0x1a3bc84c17b1d543), // 5^-191
    (0xe41f3d6a7377eeca, 0x20caba5f1d9e4a94), // 5^-190
    (0x8e938662882af53e, 0x547eb47b7282ee9d), // 5^-189
    (0xb23867fb2a35b28d, 0xe99e619a4f23aa44), // 5^-188
    (0xdec681f9f4c31f31, 0x6405fa00e2ec94d5), // 5^-187
    (0x8b3c113c38f9f37e, 0xde83bc408dd3dd05), // 5^-186
    (0xae0b158b4738705e, 0x9624ab50b148d446), // 5^-185
    (0xd98ddaee19068c76, 0x3badd624dd9b0958), // 5^-184
    (0x87f8a8d4cfa417c9, 0xe54ca5d70a80e5d7), // 5^-183
    (0xa9f6d30a038d1dbc, 0x5e9fcf4ccd211f4d), // 5^-182
    (0xd47487cc8470652b, 0x7647c32000696720), // 5^-181
    (0x84c8d4dfd2c63f3b, 0x29ecd9f40041e074), // 5^-180
    (0xa5fb0a17c777cf09, 0xf468107100525891), // 5^-179
    (0xcf79cc9db955c2cc, 0x7182148d4066eeb5), // 5^-178
    (0x81ac1fe293d599bf, 0xc6f14cd848405531), // 5^-177
    (0xa21727db38cb002f, 0xb8ada00e5a506a7d), // 5^-176
    (0xca9cf1d206fdc03b, 0xa6d90811f0e4851d), // 5^-175
    (0xfd442e4688bd304a, 0x908f4a166d1da664), // 5^-174
    (0x9e4a9cec15763e2e, 0x9a598e4e043287ff), // 5^-173
    (0xc5dd44271ad3cdba, 0x40eff1e1853f29fe), // 5^-172
    (0xf7549530e188c128, 0xd12bee59e68ef47d), // 5^-171
    (0x9a94dd3e8cf578b9, 0x82bb74f8301958cf), // 5^-170
    (0xc13a148e3032d6e7, 0xe36a52363c1faf02), // 5^-169
    (0xf18899b1bc3f8ca1, 0xdc44e6c3cb279ac2), // 5^-168
    (0x96f5600f15a7b7e5, 0x29ab103a5ef8c0ba), // 5^-167
    (0xbcb2b812db11a5de, 0x7415d448f6b6f0e8), // 5^-166
    (0xebdf661791d60f56, 0x111b495b3464ad22), // 5^-165
    (0x936b9fcebb25c995, 0xcab10dd900beec35), // 5^-164
    (0xb84687c269ef3bfb, 0x3d5d514f40eea743), // 5^-163
    (0xe65829b3046b0afa, 0x0cb4a5a3112a5113), // 5^-162
    (0x8ff71a0fe2c2e6dc, 0x47f0e785eaba72ac), // 5^-161
    (0xb3f4e093db73a093, 0x59ed216765690f57), // 5^-160
    (0xe0f218b8d25088b8, 0x306869c13ec3532d), // 5^-159
    (0x8c974f7383725573, 0x1e414218c73a13fc), // 5^-158
    (0xafbd2350644eeacf, 0xe5d1929ef90898fb), // 5^-157
    (0xdbac6c247d62a583, 0xdf45f746b74abf3a), // 5^-156
    (0x894bc396ce5da772, 0x6b8bba8c328eb784), // 5^-155
    (0xab9eb47c81f5114f, 0x066ea92f3f326565), // 5^-154
    (0xd686619ba27255a2, 0xc80a537b0efefebe), // 5^-153
    (0x8613fd0145877585, 0xbd06742ce95f5f37), // 5^-152
    (0xa798fc4196e952e7, 0x2c48113823b73705), // 5^-151
    (0xd17f3b51fca3a7a0, 0xf75a15862ca504c6), // 5^-150
    (0x82ef85133de648c4, 0x9a984d73dbe722fc), // 5^-149
    (0xa3ab66580d5fdaf5, 0xc13e60d0d2e0ebbb), // 5^-148
    (0xcc963fee10b7d1b3, 0x318df905079926a9), // 5^-147
    (0xffbbcfe994e5c61f, 0xfdf17746497f7053), // 5^-146
    (0x9fd561f1fd0f9bd3, 0xfeb6ea8bedefa634), // 5^-145
    (0xc7caba6e7c5382c8, 0xfe64a52ee96b8fc1), // 5^-144
    (0xf9bd690a1b68637b, 0x3dfdce7aa3c673b1), // 5^-143
    (0x9c1661a651213e2d, 0x06bea10ca65c084f), // 5^-142
    (0xc31bfa0fe5698db8, 0x486e494fcff30a63), // 5^-141
    (0xf3e2f893dec3f126, 0x5a89dba3c3efccfb), // 5^-140
    (0x986ddb5c6b3a76b7, 0xf89629465a75e01d), // 5^-139
    (0xbe89523386091465, 0xf6bbb397f1135824), // 5^-138
    (0xee2ba6c0678b597f, 0x746aa07ded582e2d), // 5^-137
    (0x94db483840b717ef, 0xa8c2a44eb4571cdd), // 5^-136
    (0xba121a4650e4ddeb, 0x92f34d62616ce414), // 5^-135
    (0xe896a0d7e51e1566, 0x77b020baf9c81d18), // 5^-134
    (0x915e2486ef32cd60, 0x0ace1474dc1d122f), // 5^-133
    (0xb5b5ada8aaff80b8, 0x0d819992132456bb), // 5^-132
    (0xe3231912d5bf60e6, 0x10e1fff697ed6c6a), // 5^-131
    (0x8df5efabc5979c8f, 0xca8d3ffa1ef463c2), // 5^-130
    (0xb1736b96b6fd83b3, 0xbd308ff8a6b17cb3), // 5^-129
    (0xddd0467c64bce4a0, 0xac7cb3f6d05ddbdf), // 5^-128
    (0x8aa22c0dbef60ee4, 0x6bcdf07a423aa96c), // 5^-127
    (0xad4ab7112eb3929d, 0x86c16c98d2c953c7), // 5^-126
    (0xd89d64d57a607744, 0xe871c7bf077ba8b8), // 5^-125
    (0x87625f056c7c4a8b, 0x11471cd764ad4973), // 5^-124
    (0xa93af6c6c79b5d2d, 0xd598e40d3dd89bd0), // 5^-123
    (0xd389b47879823479, 0x4aff1d108d4ec2c4), // 5^-122
    (0x843610cb4bf160cb, 0xcedf722a585139bb), // 5^-121
    (0xa54394fe1eedb8fe, 0xc2974eb4ee658829), // 5^-120
    (0xce947a3da6a9273e, 0x733d226229feea33), // 5^-119
    (0x811ccc668829b887, 0x0806357d5a3f5260), // 5^-118
    (0xa163ff802a3426a8, 0xca07c2dcb0cf26f8), // 5^-117
    (0xc9bcff6034c13052, 0xfc89b393dd02f0b6), // 5^-116
    (0xfc2c3f3841f17c67, 0xbbac2078d443ace3), // 5^-115
    (0x9d9ba7832936edc0, 0xd54b944b84aa4c0e), // 5^-114
    (0xc5029163f384a931, 0x0a9e795e65d4df12), // 5^-113
    (0xf64335bcf065d37d, 0x4d4617b5ff4a16d6), // 5^-112
    (0x99ea0196163fa42e, 0x504bced1bf8e4e46), // 5^-111
    (0xc06481fb9bcf8d39, 0xe45ec2862f71e1d7), // 5^-110
    (0xf07da27a82c37088, 0x5d767327bb4e5a4d), // 5^-109
    (0x964e858c91ba2655, 0x3a6a07f8d510f870), // 5^-108
    (0xbbe226efb628afea, 0x890489f70a55368c), // 5^-107
    (0xeadab0aba3b2dbe5, 0x2b45ac74ccea842f), // 5^-106
    (0x92c8ae6b464fc96f, 0x3b0b8bc90012929e), // 5^-105
    (0xb77ada0617e3bbcb, 0x09ce6ebb40173745), // 5^-104
    (0xe55990879ddcaabd, 0xcc420a6a101d0516), // 5^-103
    (0x8f57fa54c2a9eab6, 0x9fa946824a12232e), // 5^-102
    (0xb32df8e9f3546564, 0x47939822dc96abfa), // 5^-101
    (0xdff9772470297ebd, 0x59787e2b93bc56f8), // 5^-100
    (0x8bfbea76c619ef36, 0x57eb4edb3c55b65b), // 5^-99
    (0xaefae51477a06b03, 0xede622920b6b23f2), // 5^-98
    (0xdab99e59958885c4, 0xe95fab368e45ecee), // 5^-97
    (0x88b402f7fd75539b, 0x11dbcb0218ebb415), // 5^-96
    (0xaae103b5fcd2a881, 0xd652bdc29f26a11a), // 5^-95
    (0xd59944a37c0752a2, 0x4be76d3346f04960), // 5^-94
    (0x857fcae62d8493a5, 0x6f70a4400c562ddc), // 5^-93
    (0xa6dfbd9fb8e5b88e, 0xcb4ccd500f6bb953), // 5^-92
    (0xd097ad07a71f26b2, 0x7e2000a41346a7a8), // 5^-91
    (0x825ecc24c873782f, 0x8ed400668c0c28c9), // 5^-90
    (0xa2f67f2dfa90563b, 0x728900802f0f32fb), // 5^-89
    (0xcbb41ef979346bca, 0x4f2b40a03ad2ffba), // 5^-88
    (0xfea126b7d78186bc, 0xe2f610c84987bfa9), // 5^-87
    (0x9f24b832e6b0f436, 0x0dd9ca7d2df4d7ca), // 5^-86
    (0xc6ede63fa05d3143, 0x91503d1c79720dbc), // 5^-85
    (0xf8a95fcf88747d94, 0x75a44c6397ce912b), // 5^-84
    (0x9b69dbe1b548ce7c, 0xc986afbe3ee11abb), // 5^-83
    (0xc24452da229b021b, 0xfbe85badce996169), // 5^-82
    (0xf2d56790ab41c2a2, 0xfae27299423fb9c4), // 5^-81
    (0x97c560ba6b0919a5, 0xdccd879fc967d41b), // 5^-80
    (0xbdb6b8e905cb600f, 0x5400e987bbc1c921), // 5^-79
    (0xed246723473e3813, 0x290123e9aab23b69), // 5^-78
    (0x9436c0760c86e30b, 0xf9a0b6720aaf6522), // 5^-77
    (0xb94470938fa89bce, 0xf808e40e8d5b3e6a), // 5^-76
    (0xe7958cb87392c2c2, 0xb60b1d1230b20e05), // 5^-75
    (0x90bd77f3483bb9b9, 0xb1c6f22b5e6f48c3), // 5^-74
    (0xb4ecd5f01a4aa828, 0x1e38aeb6360b1af4), // 5^-73
    (0xe2280b6c20dd5232, 0x25c6da63c38de1b1), // 5^-72
    (0x8d590723948a535f, 0x579c487e5a38ad0f), // 5^-71
    (0xb0af48ec79ace837, 0x2d835a9df0c6d852), // 5^-70
    (0xdcdb1b2798182244, 0xf8e431456cf88e66), // 5^-69
    (0x8a08f0f8bf0f156b, 0x1b8e9ecb641b5900), // 5^-68
    (0xac8b2d36eed2dac5, 0xe272467e3d222f40), // 5^-67
    (0xd7adf884aa879177, 0x5b0ed81dcc6abb10), // 5^-66
    (0x86ccbb52ea94baea, 0x98e947129fc2b4ea), // 5^-65
    (0xa87fea27a539e9a5, 0x3f2398d747b36225), // 5^-64
    (0xd29fe4b18e88640e, 0x8eec7f0d19a03aae), // 5^-63
    (0x83a3eeeef9153e89, 0x1953cf68300424ad), // 5^-62
    (0xa48ceaaab75a8e2b, 0x5fa8c3423c052dd8), // 5^-61
    (0xcdb02555653131b6, 0x3792f412cb06794e), // 5^-60
    (0x808e17555f3ebf11, 0xe2bbd88bbee40bd1), // 5^-59
    (0xa0b19d2ab70e6ed6, 0x5b6aceaeae9d0ec5), // 5^-58
    (0xc8de047564d20a8b, 0xf245825a5a445276), // 5^-57
    (0xfb158592be068d2e, 0xeed6e2f0f0d56713), // 5^-56
    (0x9ced737bb6c4183d, 0x55464dd69685606c), // 5^-55
    (0xc428d05aa4751e4c, 0xaa97e14c3c26b887), // 5^-54
    (0xf53304714d9265df, 0xd53dd99f4b3066a9), // 5^-53
    (0x993fe2c6d07b7fab, 0xe546a8038efe402a), // 5^-52
    (0xbf8fdb78849a5f96, 0xde98520472bdd034), // 5^-51
    (0xef73d256a5c0f77c, 0x963e66858f6d4441), // 5^-50
    (0x95a8637627989aad, 0xdde7001379a44aa9), // 5^-49
    (0xbb127c53b17ec159, 0x5560c018580d5d53), // 5^-48
    (0xe9d71b689dde71af, 0xaab8f01e6e10b4a7), // 5^-47
    (0x9226712162ab070d, 0xcab3961304ca70e9), // 5^-46
    (0xb6b00d69bb55c8d1, 0x3d607b97c5fd0d23), // 5^-45
    (0xe45c10c42a2b3b05, 0x8cb89a7db77c506b), // 5^-44
    (0x8eb98a7a9a5b04e3, 0x77f3608e92adb243), // 5^-43
    (0xb267ed1940f1c61c, 0x55f038b237591ed4), // 5^-42
    (0xdf01e85f912e37a3, 0x6b6c46dec52f6689), // 5^-41
    (0x8b61313bbabce2c6, 0x2323ac4b3b3da016), // 5^-40
    (0xae397d8aa96c1b77, 0xabec975e0a0d081b), // 5^-39
    (0xd9c7dced53c72255, 0x96e7bd358c904a22), // 5^-38
    (0x881cea14545c7575, 0x7e50d64177da2e55), // 5^-37
    (0xaa242499697392d2, 0xdde50bd1d5d0b9ea), // 5^-36
    (0xd4ad2dbfc3d07787, 0x955e4ec64b44e865), // 5^-35
    (0x84ec3c97da624ab4, 0xbd5af13bef0b113f), // 5^-34
    (0xa6274bbdd0fadd61, 0xecb1ad8aeacdd58f), // 5^-33
    (0xcfb11ead453994ba, 0x67de18eda5814af3), // 5^-32
    (0x81ceb32c4b43fcf4, 0x80eacf948770ced8), // 5^-31
    (0xa2425ff75e14fc31, 0xa1258379a94d028e), // 5^-30
    (0xcad2f7f5359a3b3e, 0x096ee45813a04331), // 5^-29
    (0xfd87b5f28300ca0d, 0x8bca9d6e188853fd), // 5^-28
    (0x9e74d1b791e07e48, 0x775ea264cf55347e), // 5^-27
    (0xc612062576589dda, 0x95364afe032a819e), // 5^-26
    (0xf79687aed3eec551, 0x3a83ddbd83f52205), // 5^-25
    (0x9abe14cd44753b52, 0xc4926a9672793543), // 5^-24
    (0xc16d9a0095928a27, 0x75b7053c0f178294), // 5^-23
    (0xf1c90080baf72cb1, 0x5324c68b12dd6339), // 5^-22
    (0x971da05074da7bee, 0xd3f6fc16ebca5e04), // 5^-21
    (0xbce5086492111aea, 0x88f4bb1ca6bcf585), // 5^-20
    (0xec1e4a7db69561a5, 0x2b31e9e3d06c32e6), // 5^-19
    (0x9392ee8e921d5d07, 0x3aff322e62439fd0), // 5^-18
    (0xb877aa3236a4b449, 0x09befeb9fad487c3), // 5^-17
    (0xe69594bec44de15b, 0x4c2ebe687989a9b4), // 5^-16
    (0x901d7cf73ab0acd9, 0x0f9d37014bf60a11), // 5^-15
    (0xb424dc35095cd80f, 0x538484c19ef38c95), // 5^-14
    (0xe12e13424bb40e13, 0x2865a5f206b06fba), // 5^-13
    (0x8cbccc096f5088cb, 0xf93f87b7442e45d4), // 5^-12
    (0xafebff0bcb24aafe, 0xf78f69a51539d749), // 5^-11
    (0xdbe6fecebdedd5be, 0xb573440e5a884d1c), // 5^-10
    (0x89705f4136b4a597, 0x31680a88f8953031), // 5^-9
    (0xabcc77118461cefc, 0xfdc20d2b36ba7c3e), // 5^-8
    (0xd6bf94d5e57a42bc, 0x3d32907604691b4d), // 5^-7
    (0x8637bd05af6c69b5, 0xa63f9a49c2c1b110), // 5^-6
    (0xa7c5ac471b478423, 0x0fcf80dc33721d54), // 5^-5
    (0xd1b71758e219652b, 0xd3c36113404ea4a9), // 5^-4
    (0x83126e978d4fdf3b, 0x645a1cac083126ea), // 5^-3
    (0xa3d70a3d70a3d70a, 0x3d70a3d70a3d70a4), // 5^-2
    (0xcccccccccccccccc, 0xcccccccccccccccd), // 5^-1
    (0x8000000000000000, 0x0000000000000000), // 5^0
    (0xa000000000000000, 0x0000000000000000), // 5^1
    (0xc800000000000000, 0x0000000000000000), // 5^2
    (0xfa00000000000000, 0x0000000000000000), // 5^3
    (0x9c40000000000000, 0x0000000000000000), // 5^4
    (0xc350000000000000, 0x0000000000000000), // 5^5
    (0xf424000000000000, 0x0000000000000000), // 5^6
    (0x9896800000000000, 0x0000000000000000), // 5^7
    (0xbebc200000000000, 0x0000000000000000), // 5^8
    (0xee6b280000000000, 0x0000000000000000), // 5^9
    (0x9502f90000000000, 0x0000000000000000), // 5^10
    (0xba43b74000000000, 0x0000000000000000), // 5^11
    (0xe8d4a51000000000, 0x0000000000000000), // 5^12
    (0x9184e72a00000000, 0x0000000000000000), // 5^13
    (0xb5e620f480000000, 0x0000000000000000), // 5^14
    (0xe35fa931a0000000, 0x0000000000000000), // 5^15
    (0x8e1bc9bf04000000, 0x0000000000000000), // 5^16
    (0xb1a2bc2ec5000000, 0x0000000000000000), // 5^17
    (0xde0b6b3a76400000, 0x0000000000000000), // 5^18
    (0x8ac7230489e80000, 0x0000000000000000), // 5^19
    (0xad78ebc5ac620000, 0x0000000000000000), // 5^20
    (0xd8d726b7177a8000, 0x0000000000000000), // 5^21
    (0x878678326eac9000, 0x0000000000000000), // 5^22
    (0xa968163f0a57b400, 0x0000000000000000), // 5^23
    (0xd3c21bcecceda100, 0x0000000000000000), // 5^24
    (0x84595161401484a0, 0x0000000000000000), // 5^25
    (0xa56fa5b99019a5c8, 0x0000000000000000), // 5^26
    (0xcecb8f27f4200f3a, 0x0000000000000000), // 5^27
    (0x813f3978f8940984, 0x4000000000000000), // 5^28
    (0xa18f07d736b90be5, 0x5000000000000000), // 5^29
    (0xc9f2c9cd04674ede, 0xa400000000000000), // 5^30
    (0xfc6f7c4045812296, 0x4d00000000000000), // 5^31
    (0x9dc5ada82b70b59d, 0xf020000000000000), // 5^32
    (0xc5371912364ce305, 0x6c28000000000000), // 5^33
    (0xf684df56c3e01bc6, 0xc732000000000000), // 5^34
    (0x9a130b963a6c115c, 0x3c7f400000000000), // 5^35
    (0xc097ce7bc90715b3, 0x4b9f100000000000), // 5^36
    (0xf0bdc21abb48db20, 0x1e86d40000000000), // 5^37
    (0x96769950b50d88f4, 0x1314448000000000), // 5^38
    (0xbc143fa4e250eb31, 0x17d955a000000000), // 5^39
    (0xeb194f8e1ae525fd, 0x5dcfab0800000000), // 5^40
    (0x92efd1b8d0cf37be, 0x5aa1cae500000000), // 5^41
    (0xb7abc627050305ad, 0xf14a3d9e40000000), // 5^42
    (0xe596b7b0c643c719, 0x6d9ccd05d0000000), // 5^43
    (0x8f7e32ce7bea5c6f, 0xe4820023a2000000), // 5^44
    (0xb35dbf821ae4f38b, 0xdda2802c8a800000), // 5^45
    (0xe0352f62a19e306e, 0xd50b2037ad200000), // 5^46
    (0x8c213d9da502de45, 0x4526f422cc340000), // 5^47
    (0xaf298d050e4395d6, 0x9670b12b7f410000), // 5^48
    (0xdaf3f04651d47b4c, 0x3c0cdd765f114000), // 5^49
    (0x88d8762bf324cd0f, 0xa5880a69fb6ac800), // 5^50
    (0xab0e93b6efee0053, 0x8eea0d047a457a00), // 5^51
    (0xd5d238a4abe98068, 0x72a4904598d6d880), // 5^52
    (0x85a36366eb71f041, 0x47a6da2b7f864750), // 5^53
    (0xa70c3c40a64e6c51, 0x999090b65f67d924), // 5^54
    (0xd0cf4b50cfe20765, 0xfff4b4e3f741cf6d), // 5^55
    (0x82818f1281ed449f, 0xbff8f10e7a8921a5), // 5^56
    (0xa321f2d7226895c7, 0xaff72d52192b6a0e), // 5^57
    (0xcbea6f8ceb02bb39, 0x9bf4f8a69f764491), // 5^58
    (0xfee50b7025c36a08, 0x02f236d04753d5b5), // 5^59
    (0x9f4f2726179a2245, 0x01d762422c946591), // 5^60
    (0xc722f0ef9d80aad6, 0x424d3ad2b7b97ef6), // 5^61
    (0xf8ebad2b84e0d58b, 0xd2e0898765a7deb3), // 5^62
    (0x9b934c3b330c8577, 0x63cc55f49f88eb30), // 5^63
    (0xc2781f49ffcfa6d5, 0x3cbf6b71c76b25fc), // 5^64
    (0xf316271c7fc3908a, 0x8bef464e3945ef7b), // 5^65
    (0x97edd871cfda3a56, 0x97758bf0e3cbb5ad), // 5^66
    (0xbde94e8e43d0c8ec, 0x3d52eeed1cbea318), // 5^67
    (0xed63a231d4c4fb27, 0x4ca7aaa863ee4bde), // 5^68
    (0x945e455f24fb1cf8, 0x8fe8caa93e74ef6b), // 5^69
    (0xb975d6b6ee39e436, 0xb3e2fd538e122b45), // 5^70
    (0xe7d34c64a9c85d44, 0x60dbbca87196b617), // 5^71
    (0x90e40fbeea1d3a4a, 0xbc8955e946fe31ce), // 5^72
    (0xb51d13aea4a488dd, 0x6babab6398bdbe42), // 5^73
    (0xe264589a4dcdab14, 0xc696963c7eed2dd2), // 5^74
    (0x8d7eb76070a08aec, 0xfc1e1de5cf543ca3), // 5^75
    (0xb0de65388cc8ada8, 0x3b25a55f43294bcc), // 5^76
    (0xdd15fe86affad912, 0x49ef0eb713f39ebf), // 5^77
    (0x8a2dbf142dfcc7ab, 0x6e3569326c784338), // 5^78
    (0xacb92ed9397bf996, 0x49c2c37f07965405), // 5^79
    (0xd7e77a8f87daf7fb, 0xdc33745ec97be907), // 5^80
    (0x86f0ac99b4e8dafd, 0x69a028bb3ded71a4), // 5^81
    (0xa8acd7c0222311bc, 0xc40832ea0d68ce0d), // 5^82
    (0xd2d80db02aabd62b, 0xf50a3fa490c30191), // 5^83
    (0x83c7088e1aab65db, 0x792667c6da79e0fb), // 5^84
    (0xa4b8cab1a1563f52, 0x577001b891185939), // 5^85
    (0xcde6fd5e09abcf26, 0xed4c0226b55e6f87), // 5^86
    (0x80b05e5ac60b6178, 0x544f8158315b05b5), // 5^87
    (0xa0dc75f1778e39d6, 0x696361ae3db1c722), // 5^88
    (0xc913936dd571c84c, 0x03bc3a19cd1e38ea), // 5^89
    (0xfb5878494ace3a5f, 0x04ab48a04065c724), // 5^90
    (0x9d174b2dcec0e47b, 0x62eb0d64283f9c77), // 5^91
    (0xc45d1df942711d9a, 0x3ba5d0bd324f8395), // 5^92
    (0xf5746577930d6500, 0xca8f44ec7ee3647a), // 5^93
    (0x9968bf6abbe85f20, 0x7e998b13cf4e1ecc), // 5^94
    (0xbfc2ef456ae276e8, 0x9e3fedd8c321a67f), // 5^95
    (0xefb3ab16c59b14a2, 0xc5cfe94ef3ea101f), // 5^96
    (0x95d04aee3b80ece5, 0xbba1f1d158724a13), // 5^97
    (0xbb445da9ca61281f, 0x2a8a6e45ae8edc98), // 5^98
    (0xea1575143cf97226, 0xf52d09d71a3293be), // 5^99
    (0x924d692ca61be758, 0x593c2626705f9c57), // 5^100
    (0xb6e0c377cfa2e12e, 0x6f8b2fb00c77836d), // 5^101
    (0xe498f455c38b997a, 0x0b6dfb9c0f956448), // 5^102
    (0x8edf98b59a373fec, 0x4724bd4189bd5ead), // 5^103
    (0xb2977ee300c50fe7, 0x58edec91ec2cb658), // 5^104
    (0xdf3d5e9bc0f653e1, 0x2f2967b66737e3ee), // 5^105
    (0x8b865b215899f46c, 0xbd79e0d20082ee75), // 5^106
    (0xae67f1e9aec07187, 0xecd8590680a3aa12), // 5^107
    (0xda01ee641a708de9, 0xe80e6f4820cc9496), // 5^108
    (0x884134fe908658b2, 0x3109058d147fdcde), // 5^109
    (0xaa51823e34a7eede, 0xbd4b46f0599fd416), // 5^110
    (0xd4e5e2cdc1d1ea96, 0x6c9e18ac7007c91b), // 5^111
    (0x850fadc09923329e, 0x03e2cf6bc604ddb1), // 5^112
    (0xa6539930bf6bff45, 0x84db8346b786151d), // 5^113
    (0xcfe87f7cef46ff16, 0xe612641865679a64), // 5^114
    (0x81f14fae158c5f6e, 0x4fcb7e8f3f60c07f), // 5^115
    (0xa26da3999aef7749, 0xe3be5e330f38f09e), // 5^116
    (0xcb090c8001ab551c, 0x5cadf5bfd3072cc6), // 5^117
    (0xfdcb4fa002162a63, 0x73d9732fc7c8f7f7), // 5^118
    (0x9e9f11c4014dda7e, 0x2867e7fddcdd9afb), // 5^119
    (0xc646d63501a1511d, 0xb281e1fd541501b9), // 5^120
    (0xf7d88bc24209a565, 0x1f225a7ca91a4227), // 5^121
    (0x9ae757596946075f, 0x3375788de9b06959), // 5^122
    (0xc1a12d2fc3978937, 0x0052d6b1641c83af), // 5^123
    (0xf209787bb47d6b84, 0xc0678c5dbd23a49b), // 5^124
    (0x9745eb4d50ce6332, 0xf840b7ba963646e1), // 5^125
    (0xbd176620a501fbff, 0xb650e5a93bc3d899), // 5^126
    (0xec5d3fa8ce427aff, 0xa3e51f138ab4cebf), // 5^127
    (0x93ba47c980e98cdf, 0xc66f336c36b10138), // 5^128
    (0xb8a8d9bbe123f017, 0xb80b0047445d4185), // 5^129
    (0xe6d3102ad96cec1d, 0xa60dc059157491e6), // 5^130
    (0x9043ea1ac7e41392, 0x87c89837ad68db30), // 5^131
    (0xb454e4a179dd1877, 0x29babe4598c311fc), // 5^132
    (0xe16a1dc9d8545e94, 0xf4296dd6fef3d67b), // 5^133
    (0x8ce2529e2734bb1d, 0x1899e4a65f58660d), // 5^134
    (0xb01ae745b101e9e4, 0x5ec05dcff72e7f90), // 5^135
    (0xdc21a1171d42645d, 0x76707543f4fa1f74), // 5^136
    (0x899504ae72497eba, 0x6a06494a791c53a9), // 5^137
    (0xabfa45da0edbde69, 0x0487db9d17636893), // 5^138
    (0xd6f8d7509292d603, 0x45a9d2845d3c42b7), // 5^139
    (0x865b86925b9bc5c2, 0x0b8a2392ba45a9b3), // 5^140
    (0xa7f26836f282b732, 0x8e6cac7768d7141f), // 5^141
    (0xd1ef0244af2364ff, 0x3207d795430cd927), // 5^142
    (0x8335616aed761f1f, 0x7f44e6bd49e807b9), // 5^143
    (0xa402b9c5a8d3a6e7, 0x5f16206c9c6209a7), // 5^144
    (0xcd036837130890a1, 0x36dba887c37a8c10), // 5^145
    (0x802221226be55a64, 0xc2494954da2c978a), // 5^146
    (0xa02aa96b06deb0fd, 0xf2db9baa10b7bd6d), // 5^147
    (0xc83553c5c8965d3d, 0x6f92829494e5acc8), // 5^148
    (0xfa42a8b73abbf48c, 0xcb772339ba1f17fa), // 5^149
    (0x9c69a97284b578d7, 0xff2a760414536efc), // 5^150
    (0xc38413cf25e2d70d, 0xfef5138519684abb), // 5^151
    (0xf46518c2ef5b8cd1, 0x7eb258665fc25d6a), // 5^152
    (0x98bf2f79d5993802, 0xef2f773ffbd97a62), // 5^153
    (0xbeeefb584aff8603, 0xaafb550ffacfd8fb), // 5^154
    (0xeeaaba2e5dbf6784, 0x95ba2a53f983cf39), // 5^155
    (0x952ab45cfa97a0b2, 0xdd945a747bf26184), // 5^156
    (0xba756174393d88df, 0x94f971119aeef9e5), // 5^157
    (0xe912b9d1478ceb17, 0x7a37cd5601aab85e), // 5^158
    (0x91abb422ccb812ee, 0xac62e055c10ab33b), // 5^159
    (0xb616a12b7fe617aa, 0x577b986b314d600a), // 5^160
    (0xe39c49765fdf9d94, 0xed5a7e85fda0b80c), // 5^161
    (0x8e41ade9fbebc27d, 0x14588f13be847308), // 5^162
    (0xb1d219647ae6b31c, 0x596eb2d8ae258fc9), // 5^163
    (0xde469fbd99a05fe3, 0x6fca5f8ed9aef3bc), // 5^164
    (0x8aec23d680043bee, 0x25de7bb9480d5855), // 5^165
    (0xada72ccc20054ae9, 0xaf561aa79a10ae6b), // 5^166
    (0xd910f7ff28069da4, 0x1b2ba1518094da05), // 5^167
    (0x87aa9aff79042286, 0x90fb44d2f05d0843), // 5^168
    (0xa99541bf57452b28, 0x353a1607ac744a54), // 5^169
    (0xd3fa922f2d1675f2, 0x42889b8997915ce9), // 5^170
    (0x847c9b5d7c2e09b7, 0x69956135febada12), // 5^171
    (0xa59bc234db398c25, 0x43fab9837e699096), // 5^172
    (0xcf02b2c21207ef2e, 0x94f967e45e03f4bc), // 5^173
    (0x8161afb94b44f57d, 0x1d1be0eebac278f6), // 5^174
    (0xa1ba1ba79e1632dc, 0x6462d92a69731733), // 5^175
    (0xca28a291859bbf93, 0x7d7b8f7503cfdcff), // 5^176
    (0xfcb2cb35e702af78, 0x5cda735244c3d43f), // 5^177
    (0x9defbf01b061adab, 0x3a0888136afa64a8), // 5^178
    (0xc56baec21c7a1916, 0x088aaa1845b8fdd1), // 5^179
    (0xf6c69a72a3989f5b, 0x8aad549e57273d46), // 5^180
    (0x9a3c2087a63f6399, 0x36ac54e2f678864c), // 5^181
    (0xc0cb28a98fcf3c7f, 0x84576a1bb416a7de), // 5^182
    (0xf0fdf2d3f3c30b9f, 0x656d44a2a11c51d6), // 5^183
    (0x969eb7c47859e743, 0x9f644ae5a4b1b326), // 5^184
    (0xbc4665b596706114, 0x873d5d9f0dde1fef), // 5^185
    (0xeb57ff22fc0c7959, 0xa90cb506d155a7eb), // 5^186
    (0x9316ff75dd87cbd8, 0x09a7f12442d588f3), // 5^187
    (0xb7dcbf5354e9bece, 0x0c11ed6d538aeb30), // 5^188
    (0xe5d3ef282a242e81, 0x8f1668c8a86da5fb), // 5^189
    (0x8fa475791a569d10, 0xf96e017d694487bd), // 5^190
    (0xb38d92d760ec4455, 0x37c981dcc395a9ad), // 5^191
    (0xe070f78d3927556a, 0x85bbe253f47b1418), // 5^192
    (0x8c469ab843b89562, 0x93956d7478ccec8f), // 5^193
    (0xaf58416654a6babb, 0x387ac8d1970027b3), // 5^194
    (0xdb2e51bfe9d0696a, 0x06997b05fcc0319f), // 5^195
    (0x88fcf317f22241e2, 0x441fece3bdf81f04), // 5^196
    (0xab3c2fddeeaad25a, 0xd527e81cad7626c4), // 5^197
    (0xd60b3bd56a5586f1, 0x8a71e223d8d3b075), // 5^198
    (0x85c7056562757456, 0xf6872d5667844e4a), // 5^199
    (0xa738c6bebb12d16c, 0xb428f8ac016561dc), // 5^200
    (0xd106f86e69d785c7, 0xe13336d701beba53), // 5^201
    (0x82a45b450226b39c, 0xecc0024661173474), // 5^202
    (0xa34d721642b06084, 0x27f002d7f95d0191), // 5^203
    (0xcc20ce9bd35c78a5, 0x31ec038df7b441f5), // 5^204
    (0xff290242c83396ce, 0x7e67047175a15272), // 5^205
    (0x9f79a169bd203e41, 0x0f0062c6e984d387), // 5^206
    (0xc75809c42c684dd1, 0x52c07b78a3e60869), // 5^207
    (0xf92e0c3537826145, 0xa7709a56ccdf8a83), // 5^208
    (0x9bbcc7a142b17ccb, 0x88a66076400bb692), // 5^209
    (0xc2abf989935ddbfe, 0x6acff893d00ea436), // 5^210
    (0xf356f7ebf83552fe, 0x0583f6b8c4124d44), // 5^211
    (0x98165af37b2153de, 0xc3727a337a8b704b), // 5^212
    (0xbe1bf1b059e9a8d6, 0x744f18c0592e4c5d), // 5^213
    (0xeda2ee1c7064130c, 0x1162def06f79df74), // 5^214
    (0x9485d4d1c63e8be7, 0x8addcb5645ac2ba9), // 5^215
    (0xb9a74a0637ce2ee1, 0x6d953e2bd7173693), // 5^216
    (0xe8111c87c5c1ba99, 0xc8fa8db6ccdd0438), // 5^217
    (0x910ab1d4db9914a0, 0x1d9c9892400a22a3), // 5^218
    (0xb54d5e4a127f59c8, 0x2503beb6d00cab4c), // 5^219
    (0xe2a0b5dc971f303a, 0x2e44ae64840fd61e), // 5^220
    (0x8da471a9de737e24, 0x5ceaecfed289e5d3), // 5^221
    (0xb10d8e1456105dad, 0x7425a83e872c5f48), // 5^222
    (0xdd50f1996b947518, 0xd12f124e28f7771a), // 5^223
    (0x8a5296ffe33cc92f, 0x82bd6b70d99aaa70), // 5^224
    (0xace73cbfdc0bfb7b, 0x636cc64d1001550c), // 5^225
    (0xd8210befd30efa5a, 0x3c47f7e05401aa4f), // 5^226
    (0x8714a775e3e95c78, 0x65acfaec34810a72), // 5^227
    (0xa8d9d1535ce3b396, 0x7f1839a741a14d0e), // 5^228
    (0xd31045a8341ca07c, 0x1ede48111209a051), // 5^229
    (0x83ea2b892091e44d, 0x934aed0aab460433), // 5^230
    (0xa4e4b66b68b65d60, 0xf81da84d56178540), // 5^231
    (0xce1de40642e3f4b9, 0x36251260ab9d668f), // 5^232
    (0x80d2ae83e9ce78f3, 0xc1d72b7c6b42601a), // 5^233
    (0xa1075a24e4421730, 0xb24cf65b8612f820), // 5^234
    (0xc94930ae1d529cfc, 0xdee033f26797b628), // 5^235
    (0xfb9b7cd9a4a7443c, 0x169840ef017da3b2), // 5^236
    (0x9d412e0806e88aa5, 0x8e1f289560ee864f), // 5^237
    (0xc491798a08a2ad4e, 0xf1a6f2bab92a27e3), // 5^238
    (0xf5b5d7ec8acb58a2, 0xae10af696774b1dc), // 5^239
    (0x9991a6f3d6bf1765, 0xacca6da1e0a8ef2a), // 5^240
    (0xbff610b0cc6edd3f, 0x17fd090a58d32af4), // 5^241
    (0xeff394dcff8a948e, 0xddfc4b4cef07f5b1), // 5^242
    (0x95f83d0a1fb69cd9, 0x4abdaf101564f98f), // 5^243
    (0xbb764c4ca7a4440f, 0x9d6d1ad41abe37f2), // 5^244
    (0xea53df5fd18d5513, 0x84c86189216dc5ee), // 5^245
    (0x92746b9be2f8552c, 0x32fd3cf5b4e49bb5), // 5^246
    (0xb7118682dbb66a77, 0x3fbc8c33221dc2a2), // 5^247
    (0xe4d5e82392a40515, 0x0fabaf3feaa5334b), // 5^248
    (0x8f05b1163ba6832d, 0x29cb4d87f2a7400f), // 5^249
    (0xb2c71d5bca9023f8, 0x743e20e9ef511013), // 5^250
    (0xdf78e4b2bd342cf6, 0x914da9246b255417), // 5^251
    (0x8bab8eefb6409c1a, 0x1ad089b6c2f7548f), // 5^252
    (0xae9672aba3d0c320, 0xa184ac2473b529b2), // 5^253
    (0xda3c0f568cc4f3e8, 0xc9e5d72d90a2741f), // 5^254
    (0x8865899617fb1871, 0x7e2fa67c7a658893), // 5^255
    (0xaa7eebfb9df9de8d, 0xddbb901b98feeab8), // 5^256
    (0xd51ea6fa85785631, 0x552a74227f3ea566), // 5^257
    (0x8533285c936b35de, 0xd53a88958f872760), // 5^258
    (0xa67ff273b8460356, 0x8a892abaf368f138), // 5^259
    (0xd01fef10a657842c, 0x2d2b7569b0432d86), // 5^260
    (0x8213f56a67f6b29b, 0x9c3b29620e29fc74), // 5^261
    (0xa298f2c501f45f42, 0x8349f3ba91b47b90), // 5^262
    (0xcb3f2f7642717713, 0x241c70a936219a74), // 5^263
    (0xfe0efb53d30dd4d7, 0xed238cd383aa0111), // 5^264
    (0x9ec95d1463e8a506, 0xf4363804324a40ab), // 5^265
    (0xc67bb4597ce2ce48, 0xb143c6053edcd0d6), // 5^266
    (0xf81aa16fdc1b81da, 0xdd94b7868e94050b), // 5^267
    (0x9b10a4e5e9913128, 0xca7cf2b4191c8327), // 5^268
    (0xc1d4ce1f63f57d72, 0xfd1c2f611f63a3f1), // 5^269
    (0xf24a01a73cf2dccf, 0xbc633b39673c8ced), // 5^270
    (0x976e41088617ca01, 0xd5be0503e085d814), // 5^271
    (0xbd49d14aa79dbc82, 0x4b2d8644d8a74e19), // 5^272
    (0xec9c459d51852ba2, 0xddf8e7d60ed1219f), // 5^273
    (0x93e1ab8252f33b45, 0xcabb90e5c942b504), // 5^274
    (0xb8da1662e7b00a17, 0x3d6a751f3b936244), // 5^275
    (0xe7109bfba19c0c9d, 0x0cc512670a783ad5), // 5^276
    (0x906a617d450187e2, 0x27fb2b80668b24c6), // 5^277
    (0xb484f9dc9641e9da, 0xb1f9f660802dedf7), // 5^278
    (0xe1a63853bbd26451, 0x5e7873f8a0396974), // 5^279
    (0x8d07e33455637eb2, 0xdb0b487b6423e1e9), // 5^280
    (0xb049dc016abc5e5f, 0x91ce1a9a3d2cda63), // 5^281
    (0xdc5c5301c56b75f7, 0x7641a140cc7810fc), // 5^282
    (0x89b9b3e11b6329ba, 0xa9e904c87fcb0a9e), // 5^283
    (0xac2820d9623bf429, 0x546345fa9fbdcd45), // 5^284
    (0xd732290fbacaf133, 0xa97c177947ad4096), // 5^285
    (0x867f59a9d4bed6c0, 0x49ed8eabcccc485e), // 5^286
    (0xa81f301449ee8c70, 0x5c68f256bfff5a75), // 5^287
    (0xd226fc195c6a2f8c, 0x73832eec6fff3112), // 5^288
    (0x83585d8fd9c25db7, 0xc831fd53c5ff7eac), // 5^289
    (0xa42e74f3d032f525, 0xba3e7ca8b77f5e56), // 5^290
    (0xcd3a1230c43fb26f, 0x28ce1bd2e55f35ec), // 5^291
    (0x80444b5e7aa7cf85, 0x7980d163cf5b81b4), // 5^292
    (0xa0555e361951c366, 0xd7e105bcc3326220), // 5^293
    (0xc86ab5c39fa63440, 0x8dd9472bf3fefaa8), // 5^294
    (0xfa856334878fc150, 0xb14f98f6f0feb952), // 5^295
    (0x9c935e00d4b9d8d2, 0x6ed1bf9a569f33d4), // 5^296
    (0xc3b8358109e84f07, 0x0a862f80ec4700c9), // 5^297
    (0xf4a642e14c6262c8, 0xcd27bb612758c0fb), // 5^298
    (0x98e7e9cccfbd7dbd, 0x8038d51cb897789d), // 5^299
    (0xbf21e44003acdd2c, 0xe0470a63e6bd56c4), // 5^300
    (0xeeea5d5004981478, 0x1858ccfce06cac75), // 5^301
    (0x95527a5202df0ccb, 0x0f37801e0c43ebc9), // 5^302
    (0xbaa718e68396cffd, 0xd30560258f54e6bb), // 5^303
    (0xe950df20247c83fd, 0x47c6b82ef32a206a), // 5^304
    (0x91d28b7416cdd27e, 0x4cdc331d57fa5442), // 5^305
    (0xb6472e511c81471d, 0xe0133fe4adf8e953), // 5^306
    (0xe3d8f9e563a198e5, 0x58180fddd97723a7), // 5^307
    (0x8e679c2f5e44ff8f, 0x570f09eaa7ea7649), // 5^308
    (0xb201833b35d63f73, 0x2cd2cc6551e513db), // 5^309
    (0xde81e40a034bcf4f, 0xf8077f7ea65e58d2), // 5^310
    (0x8b112e86420f6191, 0xfb04afaf27faf783), // 5^311
    (0xadd57a27d29339f6, 0x79c5db9af1f9b564), // 5^312
    (0xd94ad8b1c7380874, 0x18375281ae7822bd), // 5^313
    (0x87cec76f1c830548, 0x8f2293910d0b15b6), // 5^314
    (0xa9c2794ae3a3c69a, 0xb2eb3875504ddb23), // 5^315
    (0xd433179d9c8cb841, 0x5fa60692a46151ec), // 5^316
    (0x849feec281d7f328, 0xdbc7c41ba6bcd334), // 5^317
    (0xa5c7ea73224deff3, 0x12b9b522906c0801), // 5^318
    (0xcf39e50feae16bef, 0xd768226b34870a01), // 5^319
    (0x81842f29f2cce375, 0xe6a1158300d46641), // 5^320
    (0xa1e53af46f801c53, 0x60495ae3c1097fd1), // 5^321
    (0xca5e89b18b602368, 0x385bb19cb14bdfc5), // 5^322
    (0xfcf62c1dee382c42, 0x46729e03dd9ed7b6), // 5^323
    (0x9e19db92b4e31ba9, 0x6c07a2c26a8346d2), // 5^324
    (0xc5a05277621be293, 0xc7098b7305241886), // 5^325
    (0xf70867153aa2db38, 0xb8cbee4fc66d1ea8), // 5^326
];
