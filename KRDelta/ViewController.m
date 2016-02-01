//
//  ViewController.m
//  KRDelta
//
//  Created by Kalvar Lin on 2015/11/4.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "ViewController.h"
#import "KRDelta.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    KRDelta *_delta         = [KRDelta sharedDelta];
    _delta.activeFunction   = KRDeltaActiveFunctionTanh;
    _delta.learningRate     = 0.8f;
    _delta.convergenceValue = 0.001f;
    _delta.maxIteration     = 1000;
    [_delta addPatterns:@[@1.0f, @-2.0f, @0.0f, @-1.0f] target:-1.0f];
    [_delta addPatterns:@[@0.0f, @1.5f, @-0.5f, @-1.0f] target:1.0f];
    [_delta setupWeights:@[@1.0f, @-1.0f, @0.0f, @0.5f]];
    //[_delta randomWeights];
    [_delta trainingWithIteration:^(NSInteger iteration, NSArray *weights) {
        NSLog(@"Doing %li iteration : %@", iteration, weights);
    } completion:^(BOOL success, NSArray *weights, NSInteger totalIteration) {
        NSLog(@"Done %li iteration : %@", totalIteration, weights);
        [_delta directOutputByPatterns:@[@1.0f, @-2.0f, @0.0f, @-1.0f] completion:^(NSArray *outputs) {
            NSLog(@"Direct Output : %@", outputs);
        }];
    }];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
