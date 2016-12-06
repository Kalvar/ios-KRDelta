## About

KRDelta (it just a neuron) implemented by Delta Learning Method of Machine Learning that is a supervisor and gradient method. This classic algorithm could do micro-analysis on mobile as well.

#### Podfile

```ruby
platform :ios, '8.0'
pod 'KRDelta', '~> 1.2.0'
```

## How To Get Started

#### Import
``` objective-c
#import "KRDelta.h"
```

#### Normal Case
``` objective-c
KRDelta *delta         = [KRDelta sharedDelta];
delta.activeFunction   = KRDeltaActiveFunctionTanh;
delta.learningRate     = 0.8f;
delta.convergenceValue = 0.001f;
delta.maxIteration     = 1000;
[delta addPatterns:@[@1.0f, @-2.0f, @0.0f, @-1.0f] target:-1.0f];
[delta addPatterns:@[@0.0f, @1.5f, @-0.5f, @-1.0f] target:1.0f];
[delta setupRandomMin:-0.5f max:0.5f];
[delta randomWeights];
[delta trainingWithIteration:^(NSInteger iteration, NSArray *weights) {
    NSLog(@"Doing %li iteration : %@", iteration, weights);
} completion:^(BOOL success, NSArray *weights, NSInteger totalIteration) {
    NSLog(@"Done %li iteration : %@", totalIteration, weights);
}];
```

#### Saving & Fetching Trained Neuron
If neuron has finished training that we could save it through KRDeltaFetcher in completion block or anywhere.
``` objective-c
// Saving
[delta trainingWithCompletion:^(BOOL success, NSArray *weights, NSInteger totalIteration) {
	[[KRDeltaFetcher sharedFetcher] save:delta forKey:@"A1"];    
}];
// Fetching
KRDelta *trainedDelta = [[KRDeltaFetcher sharedFetcher] objectForKey:@"A1"];
```

#### Setting Weights by Yourself
``` objective-c
[delta setupWeights:@[@1.0f, @-1.0f, @0.0f, @0.5f]];
```

## Version

V1.2.0

## LICENSE

MIT.

