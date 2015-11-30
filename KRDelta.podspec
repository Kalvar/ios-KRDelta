Pod::Spec.new do |s|
  s.name         = "KRDelta"
  s.version      = "1.1.0"
  s.summary      = "KRDelta is implemented by Delta Learning Method in Machine Learning that is also a supervisor and used gradient method to find out the best solution."
  s.description  = <<-DESC
                   Flexible using this method with neural network and classification methods could combine together.
                   DESC
  s.homepage     = "https://github.com/Kalvar/ios-KRDelta"
  s.license      = { :type => 'MIT', :file => 'LICENSE' }
  s.author       = { "Kalvar Lin" => "ilovekalvar@gmail.com" }
  s.social_media_url = "https://twitter.com/ilovekalvar"
  s.source       = { :git => "https://github.com/Kalvar/ios-KRDelta.git", :tag => s.version.to_s }
  s.platform     = :ios, '7.0'
  s.requires_arc = true
  s.public_header_files = 'ML/*.h'
  s.source_files = 'ML/*.{h,m}'
  s.frameworks   = 'Foundation'
end 