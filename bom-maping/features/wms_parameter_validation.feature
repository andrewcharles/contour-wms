Feature: WMS Parameter Validation
  In order to ensure WMS compliance
  As a User
  I want to get meaningful xml exceptions when I summit an invalid request
  
  Scenario Outline: Missing parameters
    Given The parameter <parameter> is missing   
    When I submit the request 
    Then it should return a "ServiceException" error with code "MissingParameter"
      And the message should contain <parameter>
      
  Examples:
    | parameter |
    | request   |
    | bbox      |
    # | crs       | crs parameter is missing        |
    # | width     | width parameter is missing      |
    # | height    | height parameter is missing     |
    # | format    | format parameter is missing     |
    # | version   | version parameter is missing    |
    # | layers    | layers parameter is missing     |
    # | styles    | styles parameter is missing     |
    

  Scenario: request parameter is not supported
    Given The value of "request" parameter is "GetCrazy"
    When I submit the request 
    Then it should return a "ServiceException" error with code "OperationNotSupported"
    
  Scenario Outline: bbox parameter doesn't have exactly 4 values
      Given The value of "bbox" parameter is <value>
      When I submit the request 
      Then it should return a "ServiceException" error with code "MissingDimension"
        And the message should contain "Missing value in Bounding Box"
        
    Examples:
      | value                 |
      |""                     |
      |"-180.0"               |
      |"-180.0,-90.0,180"     |

  # Scenario Outline: bbox parameter has non float values
  #   Given The value of "bbox" parameter is <value>
  #   When I submit the request 
  #   Then it should return a "ServiceException" error with code "InvalidDimensionValue"
  #     And the message should contain "Invalid bbox parameter: all values should be float"
  #     
  # Examples:
  #   | value               |
  #   |"a,-90.0,180,90"     |
  #   |"-180.0,a,180,90"    |
  #   |"-180.0,-90.0,a,90"  |
  #   |"-180.0,-90.0,180,a" |
  #   
  # Scenario Outline: width or height parameter are invalid
  #   Given The parameter <parameter> is <value>
  #   When I submit the request
  #   Then it should return a "WMSArgumentError" error
  #     And the message should contain "width and height must be positive numbers greater than 0"
  #     
  # Examples:
  #   | parameter | value | 
  #   | width     | -1    | 
  #   | height    | -1    | 
  #   | width     | 0     | 
  #   | height    | 0     | 
  #   | width     | big   | 
  #   | height    | big   |
  #   
  # 
  # Scenario: format parameter is not supported
  #   Given The parameter "format" is "image/mp3"
  #   When I submit the request 
  #   Then it should return a "WMSArgumentError" error
  #     And the message should contain "Invalid format parameter: format not supported"  
  # 
  # Scenario: styles parameter is invalid
  #   Given The parameter "styles" is "blah"
  #   When I submit the request 
  #   Then it should return a "WMSArgumentError" error
  #     And the message should contain "Invalid styles parameter: style not supported"
        
  # TODO find out default crs 
  # Scenario Outline: bbox parameter has incorrect values for  
  #       Given The parameter "bbox" is <value>
  #       When I submit the request 
  #       Then it should return a "WMSArgumentError" error
  #         And the message should contain "Invalid bbox parameter: minx and miny must be smaller than maxx and maxy"