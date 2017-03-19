//
//  KRDeltaPartial.h
//  KRDelta
//
//  Created by Kalvar Lin on 2017/3/17.
//  Copyright © 2017年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRDeltaPartial : NSObject

- (double)tanh:(double)x slope:(double)slope;
- (double)sigmoid:(double)x slope:(double)slope;
- (double)rbf:(double)x sigma:(double)sigma;
- (double)sgn:(double)x;
- (double)reLU:(double)x;
- (double)leakyReLU:(double)x;
- (double)eLU:(double)x;

@end
