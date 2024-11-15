﻿///<summary>
/// Xsens Data Packet represents the data comming from the network stream.
///</summary>
///<version>
/// 0.1, 2013.03.12 by Peter Heinen
/// 1.0, 2013.05.14 by Attila Odry, Daniël van Os
///</version>
///<remarks>
/// Copyright (c) 2013, Xsens Technologies B.V.
/// All rights reserved.
/// 
/// Redistribution and use in source and binary forms, with or without modification,
/// are permitted provided that the following conditions are met:
/// 
///  - Redistributions of source code must retain the above copyright notice, 
///        this list of conditions and the following disclaimer.
///  - Redistributions in binary form must reproduce the above copyright notice, 
///    this list of conditions and the following disclaimer in the documentation 
///    and/or other materials provided with the distribution.
/// 
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
/// AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
/// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
/// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
/// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
/// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
/// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///</remarks>
///
using System;
using System.Linq;
using UnityEngine;
using System.IO;

namespace Movella.Xsens
{
    /// <summary>
    /// This class represents an Xsens data packet.
    /// The data contains two parts, the header and the payload.
    /// The header holds information about the type of message, while the payload contains the actual data of the message.
    /// </summary>
    abstract class XsDataPacket
    {
        /// <summary>
        /// Enum for identify message types
        /// </summary>
        enum MessageType
        {
            Invalid = 0,
            PoseDataYup = 1,
            PoseDataZup = 2,
            PoseDataMarker = 3,
            MGTag = 4,
            PosDataUnity = 5,
            ScaleInfo = 10,
            PropInfo = 11,
            MetaData = 12,
            VelocityData = 21,
            COMData = 24
        }

        protected XsMvnPose pose;

        private Vector3[] convertedData; // List of vectors converted from new data packet

        protected abstract Vector3[] returnConvertedData();
        /// <summary>
        /// Parses the payload depends on the current network mode.
        /// </summary>
        /// <param name='startPoint'>
        /// Start point in the data array.
        /// </param>
        protected abstract double[] parsePayload(BinaryReader br, int segmentCount, int packetType);


        /// <summary>
        /// Initializes a new instance of the <see cref="xsens.XsDataPacket"/> class.
        /// </summary>
        /// <param name='readData'>
        /// Create the packed from this byte array.
        /// </param>
        public XsDataPacket(byte[] readData)
        {
            using (BinaryReader br = new BinaryReader(new MemoryStream(readData)))
            {
                int[] headerData = parseHeader(br);

                // Calls the correct classes parsePayload by itself (inheritance)
                double[] payloadData = parsePayload(br, headerData[3], headerData[0]);

                if (headerData[5] == 0){ 
                    // If we are using an MVN version below 2019
                    pose = new XsMvnPose(headerData[3]);
                }
                else{
                    // If we are using MVN 2019 and above
                    pose = new XsMvnPose(headerData[5], headerData[7], headerData[6]);
                }

                // Try to create a new pose with the data that was send
                pose.createPose(payloadData);

                convertedData = returnConvertedData();
            }
        }

        /// <summary>
        /// Gets the pose.
        /// </summary>
        /// <returns> The pose or null if there is no valid pose. </returns>
        public XsMvnPose getPose() => pose;

        /// <summary>
        /// Gets the converted data from a new data packet.
        /// </summary>
        /// <returns> List of vectors from the converted data.</returns>
        public Vector3[] getConvertedData() => convertedData;

        /// <summary>
        /// Parses the header.
        /// </summary>
        private int[] parseHeader(BinaryReader br)
        {
            byte[] mvnId = new byte[] { 0x4D, 0x58, 0x54, 0x50 };
            int[] headerData = new int[8];

            // First verify if the data is valid MVN data.
            if (mvnId.SequenceEqual(br.ReadBytes(4)))
            {
                headerData[0] = convertMessageType(br.ReadBytes(2)); // Message type
                headerData[1] = convert32BitInt(br.ReadBytes(4));    // Sample Counter
                headerData[2] = br.ReadByte();                       // Datagram Counter
                headerData[3] = br.ReadByte();                       // Number of items
                br.BaseStream.Position += 4;                         // Time stamp which we don't need, so we step over it
                headerData[4] = br.ReadByte();                       // Avatar ID
                headerData[5] = br.ReadByte();                       // Number of Body Segments
                headerData[6] = br.ReadByte();                       // Number of Props
                headerData[7] = br.ReadByte();                       // Number of Finger Segments
                br.BaseStream.Position += 4;                         // We also skip the next 7 empty bytes, this way the position is correct for the payload data
            }
            return headerData;
        }

        /// <summary>
        /// Converts the type of the message from string to int.
        /// </summary>
        /// <param name='incomingByteArray'> Incoming byte array. </param>
        /// <returns> The message type as integer. </returns>
        protected int convertMessageType(byte[] incomingByteArray)
        {
            int id = (int)MessageType.Invalid;
            if (incomingByteArray.Count() == 2)
            {
                id = (incomingByteArray[0] - 0x30) * 10;
                id += (incomingByteArray[1] - 0x30);
            }
            return id;
        }

        /// <summary>
        /// Since the binary reader is small endian, and the data from the packet is big endian we need to convert the data. This is done here, and simply puts the reverse data into a temp buffer and the memorystream and binaryreader make an integer of the data.
        /// </summary>
        protected int convert32BitInt(byte[] incomingByteArray)
        {
            byte[] tempByteArray = new byte[4];

            if (incomingByteArray.Count() >= 4)
            {
                tempByteArray[0] = incomingByteArray[3];
                tempByteArray[1] = incomingByteArray[2];
                tempByteArray[2] = incomingByteArray[1];
                tempByteArray[3] = incomingByteArray[0];
            }
            else
                Debug.LogWarning("[xsens] invalid Int data size:" + incomingByteArray.Count());

            return BitConverter.ToInt32(tempByteArray, 0);
        }

        /// <summary>
        /// Since the binary reader is small endian, and the data from the packet is big endian we need to convert the data. This is done here, and simply puts the reverse data into a temp buffer and the memorystream and binaryreader make an float of the data.
        /// </summary>
        protected double convert32BitFloat(byte[] incomingByteArray)
        {
            byte[] tempByteArray = new byte[4];

            if (incomingByteArray.Count() >= 4)
            {
                tempByteArray[0] = incomingByteArray[3];
                tempByteArray[1] = incomingByteArray[2];
                tempByteArray[2] = incomingByteArray[1];
                tempByteArray[3] = incomingByteArray[0];
            }
            else
                Debug.LogWarning("[xsens] invalid Float data size:" + incomingByteArray.Count());

            return BitConverter.ToSingle(tempByteArray, 0);
        }
    }
}