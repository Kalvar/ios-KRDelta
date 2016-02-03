## About

KRDelta is implemented by Delta Learning Method in Machine Learning that is also a supervisor and used gradient method to find out the best solution.

#### Podfile

```ruby
platform :ios, '7.0'
pod "KRDelta", "~> 1.1.0"
```

## How To Get Started

#### Import
``` objective-c
#import "KRDelta.h"
```

#### Sample
``` objective-c
KRDelta *_delta         = [KRDelta sharedDelta];
_delta.activeFunction   = KRDeltaActiveFunctionByTanh;
_delta.learningRate     = 0.8f;
_delta.convergenceValue = 0.001f;
_delta.maxIteration     = 1000;
[_delta addPatterns:@[@1.0f, @-2.0f, @0.0f, @-1.0f] target:-1.0f];
[_delta addPatterns:@[@0.0f, @1.5f, @-0.5f, @-1.0f] target:1.0f];
//[_delta setupWeights:@[@1.0f, @-1.0f, @0.0f, @0.5f]];
//[_delta setupRandomMin:-0.5f max:0.5f];
[_delta randomWeights];
[_delta trainingWithIteration:^(NSInteger iteration, NSArray *weights) {
    NSLog(@"Doing %li iteration : %@", iteration, weights);
} completion:^(BOOL success, NSArray *weights, NSInteger totalIteration) {
    NSLog(@"Done %li iteration : %@", totalIteration, weights);
    [_delta directOutputByPatterns:@[@1.0f, @-2.0f, @0.0f, @-1.0f] completion:^(NSArray *outputs) {
        NSLog(@"Direct Output : %@", outputs);
    }];
}];
```

## Version

V1.1.1

## LICENSE

MIT.

