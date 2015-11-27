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
    
    KRDelta *_delta       = [KRDelta sharedDelta];
    _delta.activeFunction = KRDeltaActiveFunctionByTanh;
    _delta.learningRate   = 1.0f;
    [_delta addPatterns:@[@1.0f, @-2.0f, @0.0f, @-1.0f] target:-1.0f];
    [_delta addPatterns:@[@0.0f, @1.5f, @-0.5f, @-1.0f] target:-1.0f];
    [_delta setWeights:@[@1.0f, @-1.0f, @0.0f, @0.5f]];
    [_delta trainingWithCompletion:^(BOOL success, NSArray *weights, NSInteger totalIteration) {
        NSLog(@"%li iteration : %@", totalIteration, weights);
    }];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
