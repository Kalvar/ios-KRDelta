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
    
    KRDelta *delta         = [KRDelta sharedDelta];
    delta.activeFunction   = KRDeltaActivationTanh;
    delta.learningRate     = 0.8f;
    delta.convergenceValue = 0.001f;
    delta.maxIteration     = 100;
    [delta addPatterns:@[@1.0f, @-2.0f, @0.0f, @-1.0f] target:-1.0f];
    [delta addPatterns:@[@0.0f, @1.5f, @-0.5f, @-1.0f] target:1.0f];
    
    [delta setupWeights:@[@1.0f, @-1.0f, @0.0f, @0.5f]];
    //[delta setupRandomMin:-0.5f max:0.5f];
    //[delta randomWeights];
    
    [delta trainingWithIteration:^(NSInteger iteration, NSArray *weights) {
        NSLog(@"Doing %li iteration : %@", iteration, weights);
    } completion:^(BOOL success, NSArray *weights, NSInteger totalIteration) {
        NSLog(@"Done %li iteration : %@", totalIteration, weights);
        //[[KRDeltaFetcher sharedFetcher] save:delta forKey:@"A1"];
        [delta directOutputByPatterns:@[@1.0f, @-2.0f, @0.0f, @-1.0f] completion:^(NSArray *outputs) {
            NSLog(@"Direct Output : %@", outputs);
        }];
    }];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
